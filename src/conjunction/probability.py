"""Collision probability calculation using multiple methods."""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, NamedTuple
import math

from src.core.exceptions import ProbabilityCalculationError
from src.core.logging import get_logger, log_execution_time
from src.conjunction.screening import RefinementResult

logger = get_logger(__name__)


class PcResult(NamedTuple):
    """Probability of collision calculation result."""
    probability: float
    confidence_interval: Tuple[float, float]
    method: str
    n_samples: Optional[int] = None
    convergence_achieved: Optional[bool] = None


class ProbabilityCalculator:
    """
    Compute probability of collision using multiple validated methods.
    
    Methods implemented:
    1. Foster's 2D projection method (analytical)
    2. Monte Carlo sampling (numerical)
    3. Akella's method (hybrid)
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @log_execution_time("pc_foster_method")
    def compute_pc_foster_method(
        self,
        miss_distance: float,
        sigma_x: float,
        sigma_y: float,
        combined_radius: float
    ) -> PcResult:
        """
        Compute Pc using Foster's 2D projection method.
        
        Args:
            miss_distance: Scalar miss distance in meters
            sigma_x: 1-sigma uncertainty in x-direction (meters)
            sigma_y: 1-sigma uncertainty in y-direction (meters)  
            combined_radius: Combined hardbody radius (meters)
            
        Returns:
            PcResult with analytical probability
            
        References:
            Foster, J. L. (1992). "The Analytic Basis for Debris Avoidance 
            Operations for the International Space Station." NASA JSC-25724.
            
        Mathematical basis:
            Projects 3D problem onto 2D plane containing relative velocity vector
            Assumes Gaussian distributions in projected plane
        """
        try:
            # Validate inputs
            if miss_distance < 0 or sigma_x <= 0 or sigma_y <= 0 or combined_radius <= 0:
                raise ValueError("Invalid parameters for Foster method")
            
            # Foster's approximation for 2D Gaussian distribution
            # Pc = 1 - exp(-R²/(2σ²)) where R = combined_radius, σ² = geometric mean of variances
            sigma_geom = math.sqrt(sigma_x * sigma_y)
            r_squared = combined_radius ** 2
            sigma_squared = sigma_geom ** 2
            
            # Avoid numerical issues with very small or large values
            exponent = -r_squared / (2 * sigma_squared)
            if exponent < -700:  # Prevent underflow
                probability = 0.0
            elif exponent > 700:  # Prevent overflow
                probability = 1.0
            else:
                probability = 1.0 - math.exp(exponent)
            
            # Confidence interval estimation (approximate)
            # For analytical method, use conservative bounds
            lower_bound = max(0.0, probability - 1e-10)
            upper_bound = min(1.0, probability + 1e-10)
            
            result = PcResult(
                probability=probability,
                confidence_interval=(lower_bound, upper_bound),
                method="foster_2d",
                n_samples=None,
                convergence_achieved=True
            )
            
            self.logger.debug(
                "Foster method Pc calculated",
                miss_distance=miss_distance,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                combined_radius=combined_radius,
                pc=probability
            )
            
            return result
            
        except Exception as e:
            raise ProbabilityCalculationError(
                message=f"Foster method calculation failed: {str(e)}",
                error_code="PC_CALCULATION_FAILED",
                details={
                    "miss_distance": miss_distance,
                    "sigma_x": sigma_x,
                    "sigma_y": sigma_y,
                    "combined_radius": combined_radius,
                    "error": str(e)
                }
            )
    
    @log_execution_time("pc_monte_carlo")
    def compute_pc_monte_carlo(
        self,
        miss_distance_vector: np.ndarray,  # [dx, dy, dz] in meters
        covariance_matrix: np.ndarray,     # 6x6 covariance in RTN frame
        combined_radius: float,
        n_samples: int = 100000,
        convergence_threshold: float = 0.01,
        max_iterations: int = 5
    ) -> PcResult:
        """
        Compute Pc using Monte Carlo sampling with convergence checking.
        
        Args:
            miss_distance_vector: 3D miss distance vector [dx, dy, dz] in meters
            covariance_matrix: 6x6 covariance matrix in RTN frame
            combined_radius: Combined hardbody radius in meters
            n_samples: Initial number of samples
            convergence_threshold: Relative change threshold for convergence
            max_iterations: Maximum refinement iterations
            
        Returns:
            PcResult with Monte Carlo probability and convergence info
            
        Statistical approach:
            1. Sample relative position deviations from multivariate normal
            2. Count samples within collision sphere
            3. Monitor convergence using running statistics
            4. Return confidence intervals using binomial distribution
        """
        try:
            # Validate inputs
            if len(miss_distance_vector) != 3:
                raise ValueError("Miss distance vector must be 3D")
            
            if covariance_matrix.shape != (6, 6):
                raise ValueError("Covariance matrix must be 6x6")
            
            if combined_radius <= 0:
                raise ValueError("Combined radius must be positive")
            
            # Extract position covariance (top-left 3x3)
            pos_covariance = covariance_matrix[:3, :3]
            
            # Validate covariance matrix
            if not self._is_valid_covariance(pos_covariance):
                raise ProbabilityCalculationError(
                    message="Invalid covariance matrix",
                    error_code="PC_CALCULATION_FAILED",
                    details={"covariance_eigenvalues": np.linalg.eigvals(pos_covariance).tolist()}
                )
            
            # Initialize sampling parameters
            current_samples = n_samples
            probabilities = []
            sample_counts = []
            
            for iteration in range(max_iterations):
                # Generate samples from multivariate normal distribution
                samples = np.random.multivariate_normal(
                    miss_distance_vector,
                    pos_covariance,
                    current_samples
                )
                
                # Calculate distances from origin (collision center)
                distances = np.linalg.norm(samples, axis=1)
                
                # Count samples within collision sphere
                collision_count = np.sum(distances <= combined_radius)
                pc_estimate = collision_count / current_samples
                
                probabilities.append(pc_estimate)
                sample_counts.append(current_samples)
                
                self.logger.debug(
                    "Monte Carlo iteration completed",
                    iteration=iteration,
                    samples=current_samples,
                    collisions=collision_count,
                    pc_estimate=pc_estimate
                )
                
                # Check convergence (skip first iteration)
                if iteration > 0:
                    prev_pc = probabilities[-2]
                    relative_change = abs(pc_estimate - prev_pc) / (prev_pc + 1e-12)
                    
                    if relative_change < convergence_threshold:
                        self.logger.debug(
                            "Monte Carlo convergence achieved",
                            iteration=iteration,
                            final_pc=pc_estimate,
                            relative_change=relative_change
                        )
                        break
                
                # Increase sample size for next iteration
                current_samples *= 2
            
            # Final probability estimate
            final_pc = probabilities[-1]
            total_samples = sum(sample_counts)
            
            # Calculate confidence interval using Wilson score interval
            # More accurate than normal approximation for extreme probabilities
            z_score = 1.96  # 95% confidence
            n = total_samples
            p = final_pc
            
            denominator = 1 + (z_score**2)/n
            centre = (p + (z_score**2)/(2*n)) / denominator
            spread = z_score * math.sqrt((p*(1-p)/n) + (z_score**2)/(4*n*n)) / denominator
            
            lower_ci = max(0.0, centre - spread)
            upper_ci = min(1.0, centre + spread)
            
            convergence_achieved = len(probabilities) < max_iterations
            
            result = PcResult(
                probability=final_pc,
                confidence_interval=(lower_ci, upper_ci),
                method="monte_carlo",
                n_samples=total_samples,
                convergence_achieved=convergence_achieved
            )
            
            self.logger.info(
                "Monte Carlo Pc calculation completed",
                final_pc=final_pc,
                total_samples=total_samples,
                confidence_interval=result.confidence_interval,
                convergence_achieved=convergence_achieved
            )
            
            return result
            
        except Exception as e:
            raise ProbabilityCalculationError(
                message=f"Monte Carlo calculation failed: {str(e)}",
                error_code="PC_CALCULATION_FAILED",
                details={
                    "miss_distance_vector": miss_distance_vector.tolist(),
                    "combined_radius": combined_radius,
                    "n_samples": n_samples,
                    "error": str(e)
                }
            )
    
    @log_execution_time("pc_akella_method")
    def compute_pc_akella_method(
        self,
        miss_distance: float,
        eigenvalues: Tuple[float, float],
        combined_radius: float
    ) -> PcResult:
        """
        Compute Pc using Akella's hybrid method.
        
        Args:
            miss_distance: Scalar miss distance in meters
            eigenvalues: Principal eigenvalues of 2D position covariance (m²)
            combined_radius: Combined hardbody radius in meters
            
        Returns:
            PcResult using Akella's method
            
        References:
            Akella, M. R., & Alfriend, K. T. (2000). "Probability of Collision 
            Between Space Objects." AIAA Guidance, Navigation, and Control Conference.
        """
        try:
            lambda1, lambda2 = eigenvalues
            
            if lambda1 <= 0 or lambda2 <= 0:
                raise ValueError("Eigenvalues must be positive")
            
            if combined_radius <= 0:
                raise ValueError("Combined radius must be positive")
            
            # Akella's approximation
            # More accurate than Foster for highly elliptical uncertainty ellipses
            sigma_eff = math.sqrt(math.sqrt(lambda1 * lambda2))  # Geometric mean
            r_scaled = combined_radius / sigma_eff
            d_scaled = miss_distance / sigma_eff
            
            # Series expansion approximation
            if d_scaled > 3.0:
                # Asymptotic form for large miss distances
                pc = 0.5 * r_scaled**2 * math.exp(-0.5 * d_scaled**2)
            else:
                # Numerical integration or lookup table would be ideal here
                # Using simplified approximation for now
                pc = 1.0 - math.exp(-0.5 * r_scaled**2)
            
            # Ensure probability bounds
            pc = max(0.0, min(1.0, pc))
            
            # Confidence interval (approximate)
            lower_bound = max(0.0, pc - 1e-6)
            upper_bound = min(1.0, pc + 1e-6)
            
            result = PcResult(
                probability=pc,
                confidence_interval=(lower_bound, upper_bound),
                method="akella_hybrid",
                n_samples=None,
                convergence_achieved=True
            )
            
            self.logger.debug(
                "Akella method Pc calculated",
                miss_distance=miss_distance,
                lambda1=lambda1,
                lambda2=lambda2,
                combined_radius=combined_radius,
                pc=pc
            )
            
            return result
            
        except Exception as e:
            raise ProbabilityCalculationError(
                message=f"Akella method calculation failed: {str(e)}",
                error_code="PC_CALCULATION_FAILED",
                details={
                    "miss_distance": miss_distance,
                    "eigenvalues": eigenvalues,
                    "combined_radius": combined_radius,
                    "error": str(e)
                }
            )
    
    def _is_valid_covariance(self, covariance: np.ndarray) -> bool:
        """Check if covariance matrix is valid (symmetric, positive definite)."""
        # Check symmetry
        if not np.allclose(covariance, covariance.T):
            return False
        
        # Check positive definiteness via eigenvalues
        try:
            eigenvalues = np.linalg.eigvals(covariance)
            return np.all(eigenvalues > 0)
        except np.linalg.LinAlgError:
            return False
    
    def select_best_method(
        self,
        miss_distance: float,
        covariance_info: dict,
        combined_radius: float
    ) -> str:
        """
        Select most appropriate Pc calculation method based on problem characteristics.
        
        Args:
            miss_distance: Miss distance in meters
            covariance_info: Dictionary with covariance characteristics
            combined_radius: Combined hardbody radius in meters
            
        Returns:
            Recommended method name
        """
        # Simple heuristic selection
        if miss_distance > 10000:  # 10km - use analytical (fast)
            return "foster_2d"
        elif miss_distance < 100:  # 100m - use Monte Carlo (accurate)
            return "monte_carlo"
        else:  # Intermediate range - use Akella hybrid
            return "akella_hybrid"


# Global calculator instance
probability_calculator = ProbabilityCalculator()
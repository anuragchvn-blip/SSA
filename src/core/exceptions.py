"""Core exceptions for the SSA conjunction analysis engine."""

from typing import Optional
from enum import Enum


class ErrorCode(Enum):
    """Standardized error codes for API responses."""
    
    # Data ingestion errors
    TLE_VALIDATION_FAILED = "TLE_VALIDATION_FAILED"
    SPACETRACK_AUTH_FAILED = "SPACETRACK_AUTH_FAILED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # Propagation errors
    PROPAGATION_FAILED = "PROPAGATION_FAILED"
    COVARIANCE_INVALID = "COVARIANCE_INVALID"
    
    # Conjunction analysis errors
    SCREENING_FAILED = "SCREENING_FAILED"
    PC_CALCULATION_FAILED = "PC_CALCULATION_FAILED"
    
    # Database errors
    DATABASE_CONNECTION_FAILED = "DATABASE_CONNECTION_FAILED"
    RECORD_NOT_FOUND = "RECORD_NOT_FOUND"
    
    # Authentication/Authorization
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
    
    # Internal errors
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


class BaseSSAException(Exception):
    """Base exception for all SSA engine errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: ErrorCode,
        details: Optional[dict] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.correlation_id = correlation_id


class DataIngestionError(BaseSSAException):
    """Raised when data ingestion fails."""
    pass


class TLEValidationError(DataIngestionError):
    """Raised when TLE validation fails."""
    pass


class RateLimitError(DataIngestionError):
    """Raised when API rate limits are exceeded."""
    pass


class PropagationError(BaseSSAException):
    """Raised when orbital propagation fails."""
    pass


class InvalidCovarianceError(PropagationError):
    """Raised when covariance matrix is invalid."""
    pass


class ConjunctionAnalysisError(BaseSSAException):
    """Raised when conjunction analysis fails."""
    pass


class ProbabilityCalculationError(ConjunctionAnalysisError):
    """Raised when Pc calculation fails."""
    pass


class DatabaseError(BaseSSAException):
    """Raised when database operations fail."""
    pass


class RecordNotFoundError(DatabaseError):
    """Raised when requested record is not found."""
    pass


class AuthenticationError(BaseSSAException):
    """Raised when authentication fails."""
    pass


class AuthorizationError(BaseSSAException):
    """Raised when authorization is denied."""
    pass
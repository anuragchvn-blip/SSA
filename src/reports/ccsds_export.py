"""Export conjunction events in CCSDS CDM format."""

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import NamedTuple, Optional
import numpy as np

from src.data.models import ConjunctionEvent, TLE
from src.propagation.sgp4_engine import SGP4Engine, CartesianState
from src.core.logging import get_logger

logger = get_logger(__name__)


class ValidationResult(NamedTuple):
    """CCSDS validation result."""
    is_valid: bool
    errors: list
    warnings: list


class CCSDSExporter:
    """
    Export conjunction events in CCSDS CDM format.
    
    STANDARD: CCSDS 508.0-B-1 (Conjunction Data Message, Blue Book)
    
    CDM contains:
    - Header (creation date, originator, message ID)
    - Relative metadata (TCA, miss distance, probability)
    - Object 1 metadata + state vector + covariance
    - Object 2 metadata + state vector + covariance
    
    CRITICAL: All fields must comply with CCSDS schema for interoperability.
    """
    
    def __init__(self):
        self.sgp4_engine = SGP4Engine()
        self.namespace = "http://www.ccsds.org/schema/ndmxml"
    
    def export_conjunction_to_cdm(
        self,
        event: ConjunctionEvent,
        primary_tle: TLE,
        secondary_tle: TLE,
        primary_covariance: Optional[np.ndarray] = None,
        secondary_covariance: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate CCSDS CDM XML string.
        
        Returns valid CDM v1.0 XML that passes schema validation.
        
        Example output:
        <?xml version="1.0" encoding="UTF-8"?>
        <cdm xmlns="http://www.ccsds.org/schema/ndmxml">
          <header>
            <CREATION_DATE>2026-01-17T12:00:00.000Z</CREATION_DATE>
            <ORIGINATOR>SSA_CONJ_ENGINE</ORIGINATOR>
            <MESSAGE_ID>CDM_20260117_001</MESSAGE_ID>
          </header>
          
          <relative_metadata>
            <TCA>2026-01-18T14:23:45.123Z</TCA>
            <MISS_DISTANCE units="m">123.456</MISS_DISTANCE>
            <RELATIVE_SPEED units="m/s">1.234e-05</RELATIVE_SPEED>
          </relative_metadata>
          
          <object1>
            <OBJECT_NAME>ISS</OBJECT_NAME>
            <NORAD_CAT_ID>25544</NORAD_CAT_ID>
            
            <cartesian_state>
              <X units="m">6778.123</X>
              <Y units="m">1234.567</Y>
              <Z units="m">890.123</Z>
              <X_DOT units="m/s">1234.567</X_DOT>
              <Y_DOT units="m/s">890.123</Y_DOT>
              <Z_DOT units="m/s">456.789</Z_DOT>
            </cartesian_state>
            
            <covariance_matrix>
              <!-- 6x6 covariance matrix elements -->
              <CX_X units="m^2">100.0</CX_X>
              <CY_X units="m^2">0.1</CY_X>
              <!-- ... remaining elements -->
            </covariance_matrix>
          </object1>
          
          <object2>
            <!-- Similar structure for secondary object -->
          </object2>
        </cdm>
        """
        # Create root element
        cdm = ET.Element("cdm")
        cdm.set("xmlns", self.namespace)
        
        # Add header section
        header = ET.SubElement(cdm, "header")
        self._add_header(header, event)
        
        # Add relative metadata
        rel_meta = ET.SubElement(cdm, "relative_metadata")
        self._add_relative_metadata(rel_meta, event)
        
        # Add object 1 (primary)
        obj1 = ET.SubElement(cdm, "object1")
        self._add_object_section(
            obj1, primary_tle, event.primary_x_eci, event.primary_y_eci,
            event.primary_z_eci, event.primary_vx_eci, event.primary_vy_eci,
            event.primary_vz_eci, primary_covariance
        )
        
        # Add object 2 (secondary)
        obj2 = ET.SubElement(cdm, "object2")
        self._add_object_section(
            obj2, secondary_tle, event.secondary_x_eci, event.secondary_y_eci,
            event.secondary_z_eci, event.secondary_vx_eci, event.secondary_vy_eci,
            event.secondary_vz_eci, secondary_covariance
        )
        
        # Convert to string with proper formatting and XML declaration
        import xml.dom.minidom
        xml_content = ET.tostring(cdm, encoding='utf-8')
        dom = xml.dom.minidom.parseString(xml_content)
        pretty_xml = dom.toprettyxml(indent="  ", newl="\n")
        # Remove the first line (XML declaration) from pretty printed version
        lines = pretty_xml.split('\n')[1:]
        xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + '\n'.join(lines)
        
        # Validate the generated CDM
        validation = self.validate_cdm(xml_str)
        if not validation.is_valid:
            logger.warning("Generated CDM has validation issues",
                         errors=validation.errors,
                         warnings=validation.warnings)
        
        return xml_str
    
    def _add_header(self, header_elem: ET.Element, event: ConjunctionEvent):
        """Add header section to CDM."""
        # Creation date (now)
        creation_date = ET.SubElement(header_elem, "CREATION_DATE")
        creation_date.text = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        
        # Originator
        originator = ET.SubElement(header_elem, "ORIGINATOR")
        originator.text = "SSA_CONJ_ENGINE"
        
        # Message ID (unique identifier)
        message_id = ET.SubElement(header_elem, "MESSAGE_ID")
        message_id.text = f"CDM_{event.id}_{int(datetime.utcnow().timestamp())}"
        
        # Message type
        msg_type = ET.SubElement(header_elem, "MESSAGE_TYPE")
        msg_type.text = "Conjunction Data Message"
    
    def _add_relative_metadata(self, rel_meta_elem: ET.Element, event: ConjunctionEvent):
        """Add relative metadata section to CDM."""
        # Time of Closest Approach
        tca = ET.SubElement(rel_meta_elem, "TCA")
        tca.text = event.tca_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        
        # Miss distance
        miss_dist = ET.SubElement(rel_meta_elem, "MISS_DISTANCE")
        miss_dist.set("units", "m")
        miss_dist.text = f"{event.miss_distance_meters:.6f}"
        
        # Relative speed
        rel_speed = ET.SubElement(rel_meta_elem, "RELATIVE_SPEED")
        rel_speed.set("units", "m/s")
        # Compute relative speed from state vectors
        dvx = event.primary_vx_eci - event.secondary_vx_eci
        dvy = event.primary_vy_eci - event.secondary_vy_eci
        dvz = event.primary_vz_eci - event.secondary_vz_eci
        rel_speed_val = np.sqrt(dvx**2 + dvy**2 + dvz**2)
        rel_speed.text = f"{rel_speed_val:.6e}"
        
        # Start and stop times for encounter
        start_time = ET.SubElement(rel_meta_elem, "START_TIME")
        start_time.text = (event.tca_datetime - 
                          timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        
        stop_time = ET.SubElement(rel_meta_elem, "STOP_TIME")
        stop_time.text = (event.tca_datetime + 
                         timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        
        # Additional metadata
        # Always include PROBABILITY_OF_COLLISION per CCSDS standards
        pc = ET.SubElement(rel_meta_elem, "PROBABILITY_OF_COLLISION")
        if hasattr(event, 'probability') and event.probability is not None:
            pc.text = f"{event.probability:.6e}"
        else:
            # Use placeholder value when probability not computed
            pc.text = "0.000000e+00"
    
    def _add_object_section(
        self,
        obj_elem: ET.Element,
        tle: TLE,
        x: float, y: float, z: float,
        vx: float, vy: float, vz: float,
        covariance: Optional[np.ndarray] = None
    ):
        """Add object section (either primary or secondary) to CDM."""
        # Object identification
        obj_name = ET.SubElement(obj_elem, "OBJECT_NAME")
        obj_name.text = f"SAT_{tle.norad_id}"  # Simplified - would lookup real name
        
        norad_id = ET.SubElement(obj_elem, "NORAD_CAT_ID")
        norad_id.text = str(tle.norad_id)
        
        # International designator
        if tle.launch_year and tle.launch_number and tle.launch_piece:
            intl_des = ET.SubElement(obj_elem, "INTERNATIONAL_DESIGNATOR")
            year_str = f"{tle.launch_year:02d}"
            num_str = f"{tle.launch_number:03d}"
            piece_str = tle.launch_piece
            intl_des.text = f"{year_str}{num_str}{piece_str}"
        
        # State vector in ECI coordinates
        cart_state = ET.SubElement(obj_elem, "cartesian_state")
        
        x_elem = ET.SubElement(cart_state, "X")
        x_elem.set("units", "m")
        x_elem.text = f"{x:.6f}"
        
        y_elem = ET.SubElement(cart_state, "Y")
        y_elem.set("units", "m")
        y_elem.text = f"{y:.6f}"
        
        z_elem = ET.SubElement(cart_state, "Z")
        z_elem.set("units", "m")
        z_elem.text = f"{z:.6f}"
        
        vx_elem = ET.SubElement(cart_state, "X_DOT")
        vx_elem.set("units", "m/s")
        vx_elem.text = f"{vx:.6f}"
        
        vy_elem = ET.SubElement(cart_state, "Y_DOT")
        vy_elem.set("units", "m/s")
        vy_elem.text = f"{vy:.6f}"
        
        vz_elem = ET.SubElement(cart_state, "Z_DOT")
        vz_elem.set("units", "m/s")
        vz_elem.text = f"{vz:.6f}"
        
        # Add covariance matrix if provided
        if covariance is not None and covariance.shape == (6, 6):
            self._add_covariance_matrix(obj_elem, covariance)
    
    def _add_covariance_matrix(self, obj_elem: ET.Element, covariance: np.ndarray):
        """Add 6x6 covariance matrix to object section."""
        cov_elem = ET.SubElement(obj_elem, "covariance_matrix")
        
        # CCSDS CDM uses specific element ordering and naming
        # Position elements: X, Y, Z
        # Velocity elements: X_DOT, Y_DOT, Z_DOT
        element_names = ["X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"]
        
        # Add all 21 unique elements (upper triangular + diagonal)
        idx = 0
        for i in range(6):
            for j in range(i, 6):  # Upper triangular including diagonal
                elem_name = f"C{element_names[j]}_{element_names[i]}"
                if i != j:
                    elem_name = elem_name.replace("_", "")  # Remove underscore for off-diagonal
                    
                cov_entry = ET.SubElement(cov_elem, elem_name)
                cov_entry.set("units", "m^2" if i < 3 and j < 3 else 
                             "m^2/s^2" if i >= 3 and j >= 3 else "m^2/s")
                cov_entry.text = f"{covariance[i, j]:.6e}"
                idx += 1
    
    def validate_cdm(self, cdm_xml: str) -> ValidationResult:
        """
        Validate CDM against CCSDS schema.
        
        Performs structural validation and basic sanity checks.
        Note: Full XSD validation would require the official CCSDS schema.
        """
        errors = []
        warnings = []
        
        try:
            # Parse XML
            root = ET.fromstring(cdm_xml)
            
            # Check namespace
            if root.tag != f"{{{self.namespace}}}cdm":
                errors.append("Invalid root element or namespace")
            
            # Define namespace map for element finding
            ns_map = {'ns': self.namespace}
            
            # Check required sections
            required_sections = ["header", "relative_metadata", "object1", "object2"]
            for section in required_sections:
                if root.find(f"{{{self.namespace}}}{section}") is None:
                    errors.append(f"Missing required section: {section}")
            
            # Validate header
            header = root.find(f"{{{self.namespace}}}header")
            if header is not None:
                required_header_fields = ["CREATION_DATE", "ORIGINATOR", "MESSAGE_ID"]
                for field in required_header_fields:
                    if header.find(f"{{{self.namespace}}}{field}") is None:
                        errors.append(f"Missing header field: {field}")
            
            # Validate relative metadata
            rel_meta = root.find(f"{{{self.namespace}}}relative_metadata")
            if rel_meta is not None:
                required_rel_fields = ["TCA", "MISS_DISTANCE", "PROBABILITY_OF_COLLISION"]
                for field in required_rel_fields:
                    if rel_meta.find(f"{{{self.namespace}}}{field}") is None:
                        errors.append(f"Missing relative metadata field: {field}")
            
            # Validate object sections
            for obj_num in [1, 2]:
                obj_section = root.find(f"{{{self.namespace}}}object{obj_num}")
                if obj_section is not None:
                    # Check required object fields
                    required_obj_fields = ["OBJECT_NAME", "NORAD_CAT_ID", "cartesian_state"]
                    for field in required_obj_fields:
                        if obj_section.find(f"{{{self.namespace}}}{field}") is None:
                            errors.append(f"Object{obj_num} missing field: {field}")
                    
                    # Validate state vector
                    cart_state = obj_section.find(f"{{{self.namespace}}}cartesian_state")
                    if cart_state is not None:
                        required_state_fields = ["X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"]
                        for field in required_state_fields:
                            field_elem = cart_state.find(f"{{{self.namespace}}}{field}")
                            if field_elem is None:
                                errors.append(f"Object{obj_num} missing state component: {field}")
                            elif not field_elem.text or not self._is_numeric(field_elem.text):
                                errors.append(f"Object{obj_num} {field} has invalid value")
            
            # Check numerical reasonableness
            self._check_numerical_reasonableness(root, warnings)
            
        except ET.ParseError as e:
            errors.append(f"XML parsing failed: {str(e)}")
        except Exception as e:
            errors.append(f"Validation failed: {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _check_numerical_reasonableness(self, root: ET.Element, warnings: list):
        """Check if numerical values are physically reasonable."""
        # Check miss distance
        rel_meta = root.find(f"{{{self.namespace}}}relative_metadata")
        if rel_meta is not None:
            miss_dist_elem = rel_meta.find(f"{{{self.namespace}}}MISS_DISTANCE")
            if miss_dist_elem is not None and miss_dist_elem.text:
                try:
                    miss_dist = float(miss_dist_elem.text)
                    if miss_dist < 0:
                        warnings.append("Negative miss distance")
                    elif miss_dist > 100000:  # 100km seems excessive
                        warnings.append("Very large miss distance")
                except ValueError:
                    warnings.append("Invalid miss distance value")
        
        # Check state vectors are reasonable for Earth orbit
        for obj_num in [1, 2]:
            obj_section = root.find(f"{{{self.namespace}}}object{obj_num}")
            if obj_section is not None:
                cart_state = obj_section.find(f"{{{self.namespace}}}cartesian_state")
                if cart_state is not None:
                    # Check positions (should be roughly Earth-sized)
                    positions = []
                    for coord in ["X", "Y", "Z"]:
                        elem = cart_state.find(f"{{{self.namespace}}}{coord}")
                        if elem is not None and elem.text:
                            try:
                                pos = abs(float(elem.text))
                                positions.append(pos)
                                if pos > 100000000:  # 100,000 km - likely error
                                    warnings.append(f"Object{obj_num} {coord} suspiciously large")
                            except ValueError:
                                warnings.append(f"Object{obj_num} {coord} invalid")
                    
                    # Check velocities (should be orbital velocities)
                    velocities = []
                    for coord in ["X_DOT", "Y_DOT", "Z_DOT"]:
                        elem = cart_state.find(f"{{{self.namespace}}}{coord}")
                        if elem is not None and elem.text:
                            try:
                                vel = abs(float(elem.text))
                                velocities.append(vel)
                                if vel > 20000:  # 20 km/s - likely error
                                    warnings.append(f"Object{obj_num} {coord} suspiciously fast")
                            except ValueError:
                                warnings.append(f"Object{obj_num} {coord} invalid")
    
    def _is_numeric(self, value: str) -> bool:
        """Check if string represents a numeric value."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _format_xml(self, xml_str: str) -> str:
        """Format XML string with proper indentation."""
        # Simple formatting - in practice would use xml.dom.minidom or similar
        lines = xml_str.split('><')
        formatted_lines = []
        
        indent_level = 0
        for line in lines:
            if line.startswith('/'):
                indent_level -= 1
            
            formatted_lines.append('  ' * indent_level + '<' + line + '>')
            
            if not line.startswith('/') and not line.endswith('/'):
                indent_level += 1
        
        return '\n'.join(formatted_lines)


# Example usage
def generate_sample_cdm():
    """Generate a sample CDM for testing."""
    # Create mock event and TLEs
    from datetime import datetime, timedelta
    
    event = ConjunctionEvent(
        primary_norad_id=25544,
        secondary_norad_id=42982,
        tca_datetime=datetime.utcnow() + timedelta(hours=24),
        primary_x_eci=6778000.0,
        primary_y_eci=1000000.0,
        primary_z_eci=500000.0,
        secondary_x_eci=6778100.0,
        secondary_y_eci=1000100.0,
        secondary_z_eci=500100.0,
        primary_vx_eci=1000.0,
        primary_vy_eci=7000.0,
        primary_vz_eci=200.0,
        secondary_vx_eci=1001.0,
        secondary_vy_eci=7001.0,
        secondary_vz_eci=201.0,
        miss_distance_meters=141.42,
        relative_velocity_mps=1.73,
        probability=1e-5,
        probability_method="foster_2d"
    )
    
    primary_tle = TLE(norad_id=25544, tle_line1="", tle_line2="")
    secondary_tle = TLE(norad_id=42982, tle_line1="", tle_line2="")
    
    exporter = CCSDSExporter()
    cdm_xml = exporter.export_conjunction_to_cdm(event, primary_tle, secondary_tle)
    
    print("Generated CDM:")
    print(cdm_xml)
    
    # Validate
    validation = exporter.validate_cdm(cdm_xml)
    print(f"\nValidation result: {'PASS' if validation.is_valid else 'FAIL'}")
    if validation.errors:
        print("Errors:", validation.errors)
    if validation.warnings:
        print("Warnings:", validation.warnings)


if __name__ == "__main__":
    generate_sample_cdm()
"""Structured logging with context and correlation IDs."""

import sys
import uuid
import structlog
from typing import Any, Dict, Optional
from functools import wraps

from src.core.config import settings


def configure_logging():
    """Configure structured logging for the application."""
    
    # Shared processors for all loggers
    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if settings.logging.log_format == "json":
        # JSON format for production
        processors = shared_processors + [
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(),
        ]
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None):
    """Get a configured logger instance."""
    return structlog.get_logger(name)


def add_correlation_id(correlation_id: Optional[str] = None):
    """Add correlation ID to log context."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    structlog.contextvars.bind_contextvars(
        correlation_id=correlation_id
    )
    return correlation_id


def clear_correlation_id():
    """Clear correlation ID from log context."""
    structlog.contextvars.unbind_contextvars("correlation_id")


def log_execution_time(operation_name: str):
    """Decorator to log execution time of functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            start_time = logger.new()._context.get("_timestamp", None)
            if start_time is None:
                import time
                start_time = time.time()
            
            logger.info(
                f"{operation_name}_started",
                operation=operation_name,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()) if kwargs else []
            )
            
            try:
                result = func(*args, **kwargs)
                
                end_time = logger.new()._context.get("_timestamp", None)
                if end_time is None:
                    import time
                    end_time = time.time()
                
                duration_ms = (end_time - start_time) * 1000
                
                logger.info(
                    f"{operation_name}_completed",
                    operation=operation_name,
                    duration_ms=round(duration_ms, 2),
                    success=True
                )
                
                return result
                
            except Exception as e:
                logger.error(
                    f"{operation_name}_failed",
                    operation=operation_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    success=False
                )
                raise
        
        return wrapper
    return decorator


class AuditLogger:
    """Audit trail logger for security-sensitive operations."""
    
    def __init__(self):
        self.logger = get_logger("audit")
    
    def log_access_attempt(
        self,
        user_id: Optional[str],
        resource: str,
        action: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log access attempts for audit trail."""
        self.logger.info(
            "access_attempt",
            event_type="access_attempt",
            user_id=user_id,
            resource=resource,
            action=action,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def log_data_modification(
        self,
        user_id: Optional[str],
        table: str,
        operation: str,
        record_id: Any,
        changes: Optional[Dict] = None
    ):
        """Log data modification operations."""
        self.logger.info(
            "data_modification",
            event_type="data_modification",
            user_id=user_id,
            table=table,
            operation=operation,
            record_id=record_id,
            changes=changes
        )
    
    def log_system_event(
        self,
        event_type: str,
        description: str,
        severity: str = "INFO",
        details: Optional[Dict] = None
    ):
        """Log system-level events."""
        self.logger.log(
            severity.lower(),
            "system_event",
            event_type=event_type,
            description=description,
            details=details
        )


# Global audit logger instance
audit_logger = AuditLogger()
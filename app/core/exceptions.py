from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError


class AgronomicValidationError(Exception):
    """Raised when an environmental or agricultural input violates physical reality."""
    def __init__(self, message: str, parameter: str = None):
        self.message = message
        self.parameter = parameter
        super().__init__(message)


async def agronomic_exception_handler(request: Request, exc: AgronomicValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "AgronomicValidationError",
            "message": exc.message,
            "parameter": exc.parameter,
            "status_code": 422
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field_loc = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        errors.append({
            "field": field_loc,
            "message": msg,
            "type": err.get("type", "validation_error")
        })
    return JSONResponse(
        status_code=422,
        content={
            "error": "InputValidationError",
            "message": "One or more input values violated physical or schema constraints.",
            "details": errors,
            "status_code": 422
        }
    )

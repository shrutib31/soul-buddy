from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config.supabase import verify_token
import logging

security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)


async def verify_supabase_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        user = await verify_token(token)
        logging.debug(f"Supabase token verified successfully for user_id: {user['id']}")
        return user
    except Exception as e:
        logging.error(f"Supabase token verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Supabase token verification failed: {str(e)}"
        )


async def optional_supabase_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security),
):
    """
    Like verify_supabase_token but returns None when no Authorization header is present.
    Used by the unified chat endpoint — incognito requests carry no token.
    """
    if credentials is None:
        return None
    token = credentials.credentials
    try:
        user = await verify_token(token)
        logging.debug(f"Supabase token verified successfully for user_id: {user['id']}")
        return user
    except Exception as e:
        logging.error(f"Supabase token verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Supabase token verification failed: {str(e)}"
        )

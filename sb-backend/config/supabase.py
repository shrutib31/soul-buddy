"""
Supabase Configuration

Handles Supabase client initialization and authentication operations.
"""

import logging
from typing import Dict, Any, Optional
from supabase import create_client, Client

from config.settings import settings

logger = logging.getLogger(__name__)

supabase_url = settings.supabase.url
supabase_service_key = settings.supabase.service_role_key
supabase_anon_key = settings.supabase.anon_key

if not supabase_url:
    raise ValueError('SUPABASE_URL is required in environment variables')

if not supabase_service_key:
    raise ValueError('SUPABASE_SERVICE_ROLE_KEY is required in environment variables')

if not supabase_anon_key:
    raise ValueError('SUPABASE_ANON_KEY is required in environment variables')


# Supabase Admin Client (Service Role)
# Use this for server-side operations that bypass RLS
supabase_admin: Client = create_client(supabase_url, supabase_service_key)

# Supabase Client (Anon Key)
# Use this for operations that respect Row Level Security
supabase: Client = create_client(supabase_url, supabase_anon_key)


async def create_user(email: str, password: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a new user with email and password

    Args:
        email: User email
        password: User password
        metadata: Additional user metadata

    Returns:
        Created user data
    """
    try:
        response = supabase_admin.auth.admin.create_user({
            'email': email,
            'password': password,
            'email_confirm': True,
            'user_metadata': metadata or {}
        })

        if hasattr(response, 'user') and response.user:
            logger.debug('[+] Supabase user created: %s', response.user.id)
            return response.user
        else:
            raise Exception('Failed to create user: No user data returned')

    except Exception as error:
        logger.debug('Create user error: %s', str(error))
        raise Exception(f'Failed to create user: {str(error)}')


async def sign_in_with_password(email: str, password: str) -> Dict[str, Any]:
    """
    Sign in user with email and password

    Args:
        email: User email
        password: User password

    Returns:
        User and session data
    """
    try:
        response = supabase.auth.sign_in_with_password({
            'email': email,
            'password': password
        })

        if hasattr(response, 'user') and hasattr(response, 'session'):
            return {
                'user': response.user,
                'session': response.session
            }
        else:
            raise Exception('Authentication failed')

    except Exception as error:
        logger.debug('Sign in error: %s', str(error))
        raise Exception(f'Authentication failed: {str(error)}')


async def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode JWT token

    Args:
        token: JWT access token

    Returns:
        Decoded user data
    """
    try:
        response = supabase_admin.auth.get_user(token)

        if hasattr(response, 'user') and response.user:
            return response.user
        else:
            raise Exception('Invalid token')

    except Exception as error:
        logger.debug('Token verification error: %s', str(error))
        raise Exception(f'Token verification failed: {str(error)}')


async def get_user_by_id(user_id: str) -> Dict[str, Any]:
    """
    Get user by ID

    Args:
        user_id: Supabase user ID

    Returns:
        User data
    """
    try:
        response = supabase_admin.auth.admin.get_user_by_id(user_id)

        if hasattr(response, 'user') and response.user:
            return response.user
        else:
            raise Exception('User not found')

    except Exception as error:
        logger.debug('Get user error: %s', str(error))
        raise Exception(f'Failed to get user: {str(error)}')


async def update_user_metadata(user_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update user metadata

    Args:
        user_id: Supabase user ID
        metadata: User metadata to update

    Returns:
        Updated user data
    """
    try:
        response = supabase_admin.auth.admin.update_user_by_id(user_id, {
            'user_metadata': metadata
        })

        if hasattr(response, 'user') and response.user:
            return response.user
        else:
            raise Exception('Failed to update user metadata')

    except Exception as error:
        logger.debug('Update user metadata error: %s', str(error))
        raise Exception(f'Failed to update user metadata: {str(error)}')


async def test_connection() -> bool:
    """
    Test Supabase connection

    Returns:
        True if connection successful
    """
    try:
        response = supabase_admin.auth.admin.list_users(page=1, per_page=1)
        logger.debug('[+] Supabase connection successful')
        return True
    except Exception as error:
        logger.debug('[!] Supabase connection failed: %s', str(error))
        raise error

#verify account status using supabase
async def verify_account_status(user_id: str) -> bool:
    """
    Mark the user's account as verified in Supabase.
    """
    try:
        response = supabase_admin.auth.admin.update_user_by_id(
            user_id,
            {
                "user_metadata": {
                    "account_status": "verified"
                }
            }
        )

        if response.user:
            logger.debug("[+] Account status verified: %s", response.user.id)
            return True

        return False

    except Exception as error:
        logger.debug("Verify account status error: %s", error)
        raise

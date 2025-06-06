from django.contrib.auth.signals import user_login_failed
from django.core.cache import cache
from django.http import HttpResponseForbidden

class LoginAttemptMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path == '/login/':
            ip = request.META.get('REMOTE_ADDR')
            attempts_key = f'login_attempts_{ip}'
            attempts = cache.get(attempts_key, 0)
            
            if attempts >= 5:  # Block after 5 failed attempts
                return HttpResponseForbidden("Too many login attempts. Please try again later.")
        
        return self.get_response(request)

def handle_failed_login(sender, credentials, **kwargs):
    ip = credentials.get('ip')
    if ip:
        attempts_key = f'login_attempts_{ip}'
        attempts = cache.get(attempts_key, 0)
        cache.set(attempts_key, attempts + 1, 300)  # Block for 5 minutes

user_login_failed.connect(handle_failed_login)

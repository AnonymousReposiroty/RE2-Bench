from django.middleware.csrf import CsrfViewMiddleware
obj = CsrfViewMiddleware(get_response={})

import os
from dash import CeleryManager, DiskcacheManager

if "REDIS_URL" in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery

    celery_app = Celery(
        __name__, broker=os.environ["REDIS_URL"], backend=os.environ["REDIS_URL"]
    )
    background_callback_manager = CeleryManager(celery_app)

else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    cache_dir = '/tmp'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache = diskcache.Cache(cache_dir)
    background_callback_manager = DiskcacheManager(cache)

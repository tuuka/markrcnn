from app import application, cache
import os

def main():
    cache.init_app(application, config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR" : "/tmp/cached"
    })

    with application.app_context():
        cache.clear()
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    main()

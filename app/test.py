import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

from app.routes import db, flask_app

with flask_app.app_context():
    db.create_all()

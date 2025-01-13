# app/models.py

from flask_bcrypt import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from app.routes import db
# If you define 'db' in another file (e.g. app/__init__.py or app/routes.py),
# be sure to import it properly. For example:
# from app.routes import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)  # If you want a username
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        """Hashes password and stores it."""
        self.password_hash = generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        """Checks hashed password."""
        return check_password_hash(self.password_hash, password)


class SavedSample(db.Model):
    __tablename__ = "saved_samples"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Replace String(255) with Text (unlimited size in Postgres)
    image_path = db.Column(db.Text, nullable=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='saved_samples')

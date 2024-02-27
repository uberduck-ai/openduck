from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from openduck_py.db import connection_string
from openduck_py.models import DBChatHistory

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = connection_string
app.config['SECRET_KEY'] = 'HfWg4eq80kzM446h'  # Needed for Flask-Admin
db = SQLAlchemy(app)


# Initialize Flask-Admin
admin = Admin(app, name='Openduck', template_mode='bootstrap3')
# Add model views
admin.add_view(ModelView(DBChatHistory, db.session))

if __name__ == "__main__":
    app.run(port="8000", debug=True)


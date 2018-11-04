import sqlalchemy as sql
from sqlalchemy import orm as orm

from .vars import CWD, DB

SqlEngine = sql.create_engine(f"sqlite:///{CWD / DB}")
Session: orm.Session = orm.sessionmaker(bind=SqlEngine)()

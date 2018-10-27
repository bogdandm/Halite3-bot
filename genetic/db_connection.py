import sqlalchemy as sql
from sqlalchemy import orm as orm

from .vars import CWD

SqlEngine = sql.create_engine(f"sqlite:///{CWD / 'genetic.db'}")
Session: orm.Session = orm.sessionmaker(bind=SqlEngine)()

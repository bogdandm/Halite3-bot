from .db_connection import SqlEngine
from .models import BotInstance, GameResult


def sync(engine):
    for table in (BotInstance, GameResult):
        if engine.dialect.has_table(engine, table.__tablename__):
            # noinspection PyUnresolvedReferences
            table.__table__.drop(engine)
            print(f"Table {table} dropped")
        table.metadata.create_all(engine)


sync(SqlEngine)

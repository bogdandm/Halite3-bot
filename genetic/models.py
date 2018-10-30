import json

import sqlalchemy as sql
import sqlalchemy.orm as orm
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class BotInstance(Base):
    __tablename__ = "bots"

    id = sql.Column(sql.Integer, primary_key=True, autoincrement=True)
    version = sql.Column(sql.String)
    _args_dict = sql.Column(sql.String)
    halite = sql.Column(sql.Integer, server_default='0')
    game2_count = sql.Column(sql.Integer, server_default='0')
    game4_count = sql.Column(sql.Integer, server_default='0')
    generation = sql.Column(sql.Integer)

    @property
    def args_dict(self):
        return json.loads(self._args_dict)

    @args_dict.setter
    def args_dict(self, value):
        self._args_dict = json.dumps(value)

    def __hash__(self):
        return self.id

    def __repr__(self):
        return f"<BotInstance#{self.id} h={self.halite}, games=({self.game2_count}, {self.game4_count}), " \
               f"gen={self.generation}, args={self._args_dict}>"


class GameResult(Base):
    __tablename__ = "games"

    id = sql.Column(sql.Integer, primary_key=True, autoincrement=True)
    timestamp = sql.Column(sql.DateTime(timezone=True), server_default=sql.func.now())
    generation = sql.Column(sql.Integer)

    bot1_id = sql.Column(sql.Integer, sql.ForeignKey("bots.id"), nullable=True)
    bot2_id = sql.Column(sql.Integer, sql.ForeignKey("bots.id"), nullable=True)
    bot3_id = sql.Column(sql.Integer, sql.ForeignKey("bots.id"), nullable=True)
    bot4_id = sql.Column(sql.Integer, sql.ForeignKey("bots.id"), nullable=True)

    bot1 = orm.relationship("BotInstance", foreign_keys=[bot1_id])
    bot2 = orm.relationship("BotInstance", foreign_keys=[bot2_id])
    bot3 = orm.relationship("BotInstance", foreign_keys=[bot3_id])
    bot4 = orm.relationship("BotInstance", foreign_keys=[bot4_id])

    def __hash__(self):
        return self.id


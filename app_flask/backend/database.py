from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# SQLite database path - local file storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data")
os.makedirs(DB_PATH, exist_ok=True)

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"sqlite:///{DB_PATH}/connect4.db"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class GameRecord(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    mode = Column(String)           # "hvh", "hva", "ava"
    player1_type = Column(String)   # "human" or agent name
    player2_type = Column(String)
    player1_config = Column(JSON, nullable=True)
    player2_config = Column(JSON, nullable=True)
    winner = Column(String, nullable=True)  # "player1", "player2", "draw"
    total_moves = Column(Integer)
    duration_seconds = Column(Float, nullable=True)
    final_board = Column(JSON)
    move_history = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    tournament_id = Column(String, nullable=True)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

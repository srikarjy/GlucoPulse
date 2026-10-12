from priorauth.bootstrap import database_url


def test_explicit_url_wins(monkeypatch):
    monkeypatch.setenv("PRIORAUTH_DATABASE_URL", "sqlite:///x.db"); monkeypatch.setenv("PRIORAUTH_DB_HOST", "h")
    assert database_url() == "sqlite:///x.db"


def test_assembled_url_escapes_password_and_requires_tls(monkeypatch):
    monkeypatch.delenv("PRIORAUTH_DATABASE_URL", raising=False)
    for k, v in dict(HOST="db.internal", USER="priorauth", PASSWORD="p@ss/w:rd#1?", NAME="priorauth").items():
        monkeypatch.setenv(f"PRIORAUTH_DB_{k}", v)
    u = database_url()
    assert u == "postgresql+psycopg://priorauth:p%40ss%2Fw%3Ard%231%3F@db.internal:5432/priorauth?sslmode=require"
    from sqlalchemy.engine import make_url
    assert make_url(u).password == "p@ss/w:rd#1?"


def test_defaults_to_local_sqlite(monkeypatch):
    for k in ("PRIORAUTH_DATABASE_URL", "PRIORAUTH_DB_HOST"):
        monkeypatch.delenv(k, raising=False)
    assert database_url().startswith("sqlite")

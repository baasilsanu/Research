import pandas as pd
from sqlalchemy import create_engine

#db params
db_username = 'postgres'
db_password = 'Simulation2024'
db_host = 'localhost'
db_port = '5432'
db_name = 'simulationdatainit'
table_name = 'simulationsiter2'

#creating connection
connection_string = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)

#getting df
query = f"SELECT * FROM {table_name}"
df = pd.read_sql(query, engine)
print(df.head())
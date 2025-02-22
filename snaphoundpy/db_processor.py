import sqlite3

class DatabaseManager:
	def __init__(self, db_info):
		self.db_name = db_info["DATABASE"]
		self.backup_db_name = db_info["BACKUP_DATABASE"]
		self.table_name = db_info["TABLE_NAME"]
		self.columns = db_info["COLUMNS"]
		self.init_database()

	def get_id(self, key):
		return self.columns[key]['index']

	def init_database(self):
		with sqlite3.connect(self.db_name) as conn:
			columns_sql = ", ".join(
				[
					f"{col['name']} {col['type'].upper()}" + 
					(" PRIMARY KEY AUTOINCREMENT" if col['name'] == "id" else " NULL") 
					for col in self.columns.values()
				]
			)
			create_table_query = f"CREATE TABLE IF NOT EXISTS {self.table_name} ({columns_sql})"
			conn.execute(create_table_query)

	def backup_database(self):
		import shutil
		shutil.copy2(self.db_name, self.backup_db_name)

	def get_columns(self):
		cursor = self.execute(f"SELECT * FROM {self.table_name}", fetch_type='cursor')
		return [description[0] for description in cursor.description]

	def execute(self, query, values=None, fetch_type='getAll'):
		conn = None
		try:
			conn = sqlite3.connect(self.db_name)
			cursor = conn.cursor()
			
			if values:
				cursor.execute(query, values)
			else:
				cursor.execute(query)

			if fetch_type == 'getAll':
				return cursor.fetchall()
			elif fetch_type == 'get':
				return cursor.fetchone()
			elif fetch_type == "lastrowid":
				return cursor.lastrowid
			elif fetch_type == 'cursor':
				return cursor
			return None
		except Exception as e:
			raise ValueError(f'Error in DatabaseManager.execute: {str(e)}')
		finally:
			if conn:
				conn.commit()
				conn.close()

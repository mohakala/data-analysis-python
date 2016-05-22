import MySQLdb

"""
http://www.tutorialspoint.com/python/python_database_access.htm
http://www.w3schools.com/sql/sql_syntax.asp
"""

# Open database connection
db = MySQLdb.connect("localhost","testuser","test123","TESTDB" )

# prepare a cursor object using cursor() method
cursor = db.cursor()

# execute SQL query using execute() method.
cursor.execute("SELECT VERSION()")

# Fetch a single row using fetchone() method.
data = cursor.fetchone()

print ("Database version : %s " % data)


# COMMANDS


"""
    SELECT - extracts data from a database
    UPDATE - updates data in a database
    DELETE - deletes data from a database
    INSERT INTO - inserts new data into a database
    CREATE DATABASE - creates a new database
    ALTER DATABASE - modifies a database
    CREATE TABLE - creates a new table
    ALTER TABLE - modifies a table
    DROP TABLE - deletes a table
    CREATE INDEX - creates an index (search key)
    DROP INDEX - deletes an index
"""

# show databases
# use testdb
# select * from employee

# INSERT
sql = "INSERT INTO EMPLOYEE(FIRST_NAME, \
       LAST_NAME, AGE, SEX, INCOME) \
       VALUES ('%s', '%s', '%d', '%c', '%d' )" % \
       ('Patty', 'Miller', 35, 'F', 4000)

try:
   # Execute the SQL command
   cursor.execute(sql)
   # Commit your changes in the database
   db.commit()
   print('Success executing SQL command') 
except:
   # Rollback in case there is any error
   db.rollback()
   print('Not success executing SQL command') 


# SELECT
sql = "SELECT * FROM EMPLOYEE" 

try:
   cursor.execute(sql)
   data = cursor.fetchall() 
   print("Data fetched:\n",data)
   print('Success executing SQL command') 
except:
   # Rollback in case there is any error
   db.rollback()
   print('Not success executing SQL command') 



# disconnect from server
db.close()

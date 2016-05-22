import MySQLdb

"""
http://www.tutorialspoint.com/python/python_database_access.htm
http://www.w3schools.com/sql/sql_syntax.asp
"""

def dbAction(sql):
   data=[]
   try:
      # Execute SQL command
      cursor.execute(sql)
      db.commit()
      data = cursor.fetchall()
      # print('Success executing SQL command') 
   except:
      # Rollback in case there is any error
      db.rollback()
      print('Not success executing SQL command') 
   return(data)



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
# select FIRST_NAME, LAST_NAME from employee;


# INSERT
sql = "INSERT INTO xEMPLOYEE(FIRST_NAME, \
       LAST_NAME, AGE, SEX, INCOME) \
       VALUES ('%s', '%s', '%d', '%c', '%d' )" % \
       ('Patty', 'Smithson', 20, 'F', 3000)
data=dbAction(sql)

# SELECT
sql = "SELECT * FROM EMPLOYEE" 
data=dbAction(sql)
print("Data fetched:\n",data)

# SELECT DISTINCT City FROM Customers;
sql = "SELECT DISTINCT FIRST_NAME FROM EMPLOYEE"
data=dbAction(sql)
print("Data fetched:\n",data)

# SELECT * FROM Customers WHERE Country='Mexico';
# http://www.w3schools.com/sql/sql_where.asp
sql = "SELECT INCOME FROM EMPLOYEE WHERE INCOME>2000"
data=dbAction(sql)
print("Data fetched:\n",data)

sql = "SELECT FIRST_NAME FROM EMPLOYEE WHERE FIRST_NAME LIKE '%o%'"
data=dbAction(sql)
print("Data fetched:\n",data)

sql = "SELECT first_name FROM EMPLOYEE WHERE SEX in ('M','F')"

sql = "SELECT * FROM employee WHERE (income BETWEEN 3000 AND 4000) \
AND NOT sex = 'M'"; 

sql = "SELECT * FROM employee WHERE sex='F' AND (age=31 OR age = 20)";


data=dbAction(sql)
print("Data fetched:\n",data)



# disconnect from server
db.close()

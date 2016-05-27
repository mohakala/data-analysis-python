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
      print('Not success executing SQL command:',sql) 
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

# Create another table ORDERS
sql = "xxxCREATE TABLE ORDERS (id INT NOT NULL PRIMARY KEY AUTO_INCREMENT, \
                            ORDERNAME CHAR(15), COUNTRY CHAR(15), \
                            MONTH INT (4), YEAR INT(4), TYPE CHAR(10), \
                            SIZE CHAR(8), COLOR CHAR (8))"
sql = "xxxINSERT INTO ORDERS (ordername, \
       month, year) \
       VALUES ('%s', '%d', '%d')" % \
       ('engine parts', 3, 2015)
sql = "xxxALTER TABLE ORDERS ADD LAST_NAME CHAR(15)"
sql = "xxxALTER TABLE ORDERS DROP COLUMN COLOR"
sql = "xxxUPDATE orders SET last_name='Moore' WHERE id=1"
sql = "xxxinsert into orders (ordername, month, last_name) \
   values ('%s', '%d', '%s')" % \
      ('watch',10,'Moore')
sql = "xxxinsert into orders (ordername, month, last_name) \
   values ('%s', '%d', '%s')" % \
      ('jewellery',12,'Hunt')
      
#data=dbAction(sql)


# INSERT INTO. Basic insertion.
sql = "xxxINSERT INTO EMPLOYEE(FIRST_NAME, \
       LAST_NAME, AGE, SEX, INCOME) \
       VALUES ('%s', '%s', '%d', '%c', '%d' )" % \
       ('Patty', 'Smithson', 20, 'F', 3000)
# data=dbAction(sql)

# Insert with id
sql = "INSERT INTO xEMPLOYEE(id, FIRST_NAME, \
       LAST_NAME, AGE, SEX, INCOME) \
       VALUES ('%d','%s', '%s', '%d', '%c', '%d' )" % \
       (3,'Roger', 'Moore', 72, 'M', 11000)


# SELECT
sql = "SELECT * FROM EMPLOYEE" 
data=dbAction(sql)
print("Data fetched:\n",data)

# SELECT DISTINCT City FROM Customers;
sql = "SELECT DISTINCT FIRST_NAME FROM EMPLOYEE"
data=dbAction(sql)
print("Data fetched:\n",data)

# SELECT
# http://www.w3schools.com/sql/sql_where.asp

sql = "SELECT INCOME FROM EMPLOYEE WHERE INCOME>2000"
data=dbAction(sql)
print("Data fetched:\n",data)

#sql = "SELECT FIRST_NAME FROM EMPLOYEE WHERE FIRST_NAME LIKE '%o%'"
sql = "SELECT FIRST_NAME FROM EMPLOYEE WHERE FIRST_NAME NOT LIKE '%o%'"

data=dbAction(sql)
print("Data fetched 1:\n",data)

sql = "SELECT first_name FROM EMPLOYEE WHERE SEX in ('M','F')"

# BETWEEN
sql = "SELECT * FROM employee WHERE (income BETWEEN 3000 AND 4000) \
AND NOT sex = 'M'"; 

sql = "SELECT * FROM employee WHERE sex='F' AND (age=31 OR age = 20)";

# SELECT and ORDER BY      # DESC == descending order
sql = "SELECT * FROM employee ORDER BY income DESC, last_name DESC"

# UPDATE A TABLE ENTRY
sql = "UPDATE employee SET first_name='Joshua', income=3100 WHERE first_name='Josh'"

# DELETE
sql = "DELETE FROM employee WHERE last_name='xxxxx' " 

# SQL SELECT TOP
sql = "SELECT * FROM employee LIMIT 3"

# WILDCARDS
sql = "SELECT FIRST_NAME FROM EMPLOYEE WHERE FIRST_NAME LIKE '_oger'"
sql = "SELECT FIRST_NAME FROM EMPLOYEE WHERE FIRST_NAME LIKE '%ger'"
# But the following didn't work ?!
# sql = "SELECT FIRST_NAME FROM EMPLOYEE WHERE FIRST_NAME LIKE '[r]%'"

# IN
sql = "SELECT * FROM employee WHERE first_name IN ('Patty','Roger','Shirley')"

# BETWEEN: note, here I not included
sql = "SELECT * FROM employee WHERE first_name BETWEEN 'A' AND 'I'"

# ALIAS
sql = "SELECT FIRST_NAME AS 'First Name', last_name as 'Last Name' FROM employee"

# ALIAS AND CONCATENATE
sql = "SELECT last_name, CONCAT(first_name,'+ ',age,'+ ',sex) AS info from employee"

# FETCH FROM TWO TABLES
sql = "SELECT o.ordername, o.month, c.last_name \
   FROM employee AS c, orders AS o \
   WHERE c.last_name='Moore' AND c.last_name='Moore'"

# KIRJANMERKKI: http://www.w3schools.com/sql/sql_alias.asp






# execute the action
data=dbAction(sql)
print("Data fetched:\n",data)

# disconnect from server
db.close()

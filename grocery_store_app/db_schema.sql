DROP DATABASE IF EXISTS grocery_store;
CREATE DATABASE grocery_store;

USE grocery_store;


CREATE TABLE products (
  product_id              INT           PRIMARY KEY		NOT NULL	AUTO_INCREMENT,
  name     				VARCHAR(50)    	NOT NULL,
  uom_id        		INT				NOT NULL,
  price_per_unit		DECIMAL(9,2)	NOT NULL,
  CONSTRAINT products_fk_term
    FOREIGN KEY (uom_id)
    REFERENCES uom (uom_id)
    ON DELETE NO ACTION
    ON UPDATE RESTRICT
  );

-- unit of measure - uom
CREATE TABLE uom (
  uom_id        		INT			PRIMARY KEY 	NOT NULL	AUTO_INCREMENT,
  unit_name     		VARCHAR(50)	NOT NULL
  );
  
  
  CREATE TABLE orders (
  order_id	                INT           PRIMARY KEY		NOT NULL	AUTO_INCREMENT,
  datetime   				datetime    	  NOT NULL,
  customer_name        		VARCHAR(50)	  NOT NULL,
  total_cost				DECIMAL(9,2)  NOT NULL
  );
  
  
  CREATE TABLE order_details (
  order_id			INT		NOT NULL,
  product_id        INT 	NOT NULL,
  price_per_unit    DECIMAL(9,2)	NOT NULL,
  quantity        	DOUBLE	  		NOT NULL,
  total				DECIMAL(9,2)  	NOT NULL,
	CONSTRAINT order_details_fk
    FOREIGN KEY (order_id)
    REFERENCES orders (order_id)
    ON DELETE NO ACTION
    ON UPDATE RESTRICT,
    CONSTRAINT product_id_fk
    FOREIGN KEY (product_id)
    REFERENCES products (product_id)
	ON DELETE NO ACTION
    ON UPDATE RESTRICT 
  );

INSERT INTO uom VALUES
('1', 'each'); 
INSERT INTO uom VALUES 
('2', 'kgs');

SELECT * FROM order_details;
SELECT * FROM orders;
SELECT * FROM products;
SELECT * FROM uom;


INSERT INTO products VALUES
(1, 'toothpaste', 1, 10);

INSERT INTO orders VALUES
(1, 20220512,  'Akshat', 99); 

INSERT INTO order_details VALUES
(1, 1, 10, 2, 20);

INSERT INTO products VALUES
(2, 'salt', 2, 10),
(3, 'clock', 1, 30),
(4, 'water-bottle', 1, 15),
(5, 'Pillow', 1 , 10),
(6, 'Onions', 2, 2),
(7, 'Bananas', 2, 3),
(8, 'Apples', 2, 5),
(9, 'Cilantro', 2, 3),
(10, 'Pineapple', 1, 4),
(11, 'Beer', 1, 12),
(12, 'Wine', 1, 12),
(13, 'Chips', 1, 3),
(14, 'Orange', 2, 5),
(15, 'Light Bulb', 1, 3);


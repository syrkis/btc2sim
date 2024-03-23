-- Description: This file contains the DDL for the database
PRAGMA foreign_keys = ON;

-- Drop the tables
DROP TABLE IF EXISTS btrees;
DROP TABLE IF EXISTS functions;

-- Create the tables
CREATE TABLE btrees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    tree_json TEXT NOT NULL,
    UNIQUE(name)
);

CREATE TABLE functions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    UNIQUE(name)
);


-- index.sql
CREATE INDEX idx_btrees_name ON btrees(name);
CREATE INDEX idx_functions_name ON functions(name);

-- add some functions
INSERT INTO functions (name, description) VALUES ('add', 'Add two numbers');
INSERT INTO functions (name, description) VALUES ('subtract', 'Subtract two numbers');
INSERT INTO functions (name, description) VALUES ('multiply', 'Multiply two numbers');
INSERT INTO functions (name, description) VALUES ('divide', 'Divide two numbers');

-- add first tree
INSERT INTO btrees (name, description, tree_json) VALUES ('First Tree', 'This is the first tree', '{
    "name": "First Tree",
    "description": "This is the first tree",
    "nodes": [
        {
            "name": "Root",
            "description": "Root node",
            "children": [
                {
                    "name": "Child 1",
                    "description": "Child 1 node",
                    "children": []
                },
                {
                    "name": "Child 2",
                    "description": "Child 2 node",
                    "children": []
                }
            ]
        }
    ]
}');
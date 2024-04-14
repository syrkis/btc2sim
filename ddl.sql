-- drop table answers
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS Children;
DROP TABLE IF EXISTS Nodes;
DROP TABLE IF EXISTS Kinds;
DROP TABLE IF EXISTS Atomics;


CREATE TABLE If Not Exists Nodes (
    id INTEGER PRIMARY KEY,
    kind_id INTEGER,
    name TEXT,
    description TEXT,
    atomic_id INTEGER,
    FOREIGN KEY (atomic_id) REFERENCES Atomics(id),
    FOREIGN KEY (kind_id) REFERENCES Kinds(id)
);

CREATE TABLE If Not Exists Atomics (
    id INTEGER PRIMARY KEY,
    function TEXT NOT NULL,
    description TEXT
);

CREATE TABLE If Not Exists Kinds (
    id INTEGER PRIMARY KEY,
    name TEXT
);

CREATE TABLE If Not Exists Children (
    parent_id INTEGER,
    child_id INTEGER,
    FOREIGN KEY (parent_id) REFERENCES Nodes(id),
    FOREIGN KEY (child_id) REFERENCES Nodes(id),
    PRIMARY KEY (parent_id, child_id)
);


-- create kinds
INSERT INTO Kinds (name) VALUES ('Sequence');
INSERT INTO Kinds (name) VALUES ('Fallback');
INSERT INTO Kinds (name) VALUES ('Decorator');
INSERT INTO Kinds (name) VALUES ('Action');
INSERT INTO Kinds (name) VALUES ('Condition');

-- create atomics
INSERT INTO Atomics (function, description) VALUES ('Action1', 'Action 1');
INSERT INTO Atomics (function, description) VALUES ('Action2', 'Action 2');
INSERT INTO Atomics (function, description) VALUES ('Action3', 'Action 3');
INSERT INTO Atomics (function, description) VALUES ('Condition1', 'Condition 1');
INSERT INTO Atomics (function, description) VALUES ('Condition2', 'Condition 2');
INSERT INTO Atomics (function, description) VALUES ('Condition3', 'Condition 3');

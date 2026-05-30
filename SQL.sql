CREATE DATABASE IF NOT EXISTS logitech_phase1;
USE logitech_phase1;

-- 1. STORED PROCEDURE
-- Updates reorder-related flags for all inventory rows
DROP PROCEDURE IF EXISTS sp_update_reorder_flags;

DELIMITER $$

CREATE PROCEDURE sp_update_reorder_flags()
BEGIN
    UPDATE inventory_stock_movement_clean
    SET Reorder_Flag = CASE
            WHEN Stock_On_Hand <= Reorder_Level THEN 1
            ELSE 0
        END,
        Reorder_Flag_Check = CASE
            WHEN Stock_On_Hand <= Reorder_Level THEN 1
            ELSE 0
        END,
        Reorder_Flag_Mismatch = CASE
            WHEN
                (CASE WHEN Stock_On_Hand <= Reorder_Level THEN 1 ELSE 0 END)
                <>
                (CASE WHEN Reorder_Flag = 1 THEN 1 ELSE 0 END)
            THEN 1
            ELSE 0
        END
    WHERE Product_ID IS NOT NULL;
END $$

DELIMITER ;

-- 2. TRIGGER
-- Automatically sets reorder-related flags on new insert
DROP TRIGGER IF EXISTS trg_inventory_before_insert;

DELIMITER $$

CREATE TRIGGER trg_inventory_before_insert
BEFORE INSERT ON inventory_stock_movement_clean
FOR EACH ROW
BEGIN
    SET NEW.Reorder_Flag = CASE
        WHEN NEW.Stock_On_Hand <= NEW.Reorder_Level THEN 1
        ELSE 0
    END;

    SET NEW.Reorder_Flag_Check = CASE
        WHEN NEW.Stock_On_Hand <= NEW.Reorder_Level THEN 1
        ELSE 0
    END;

    SET NEW.Reorder_Flag_Mismatch = 0;
END $$

DELIMITER ;

-- 4. RUN PROCEDURE MANUALLY
SET SQL_SAFE_UPDATES = 0;
CALL sp_update_reorder_flags();
SET SQL_SAFE_UPDATES = 1;

-- 5. VERIFICATION QUERIES
-- Use these to show your code is working

-- Check stored procedure exists
SHOW PROCEDURE STATUS
WHERE Db = 'logitech_phase1' AND Name = 'sp_update_reorder_flags';

-- Check trigger exists
SHOW TRIGGERS FROM logitech_phase1;

-- Check event exists
SHOW EVENTS FROM logitech_phase1;

-- See sample output from inventory table
SELECT
    Product_ID,
    Stock_On_Hand,
    Reorder_Level,
    Reorder_Flag,
    Reorder_Flag_Check,
    Reorder_Flag_Mismatch
FROM inventory_stock_movement_clean
ORDER BY Product_ID
LIMIT 10;



INSERT INTO inventory_stock_movement_clean
(
    Product_ID,
    Warehouse_ID,
    Category,
    Stock_On_Hand,
    Reorder_Level,
    Reorder_Flag,
    Avg_Lead_Time_Days,
    Stockout_Days,
    Carrying_Cost_Per_Unit,
    Reorder_Flag_Check,
    Reorder_Flag_Mismatch,
    Inventory_Duplicate_Flag,
    Stock_On_Hand_Outlier,
    Reorder_Level_Outlier,
    Avg_Lead_Time_Days_Outlier,
    Stockout_Days_Outlier,
    Carrying_Cost_Per_Unit_Outlier
)
VALUES
(
    999999,
    'WH_TEST',
    'Test_Category',
    5,
    10,
    0,
    3,
    0,
    1.50,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
);

SELECT
    Product_ID,
    Stock_On_Hand,
    Reorder_Level,
    Reorder_Flag,
    Reorder_Flag_Check,
    Reorder_Flag_Mismatch
FROM inventory_stock_movement_clean
WHERE Product_ID = 999999;



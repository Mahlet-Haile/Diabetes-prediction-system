CREATE DATABASE IF NOT EXISTS diabetes_prediction CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'diabetes_user'@'localhost' IDENTIFIED BY 'diabetes_password';
GRANT ALL PRIVILEGES ON diabetes_prediction.* TO 'diabetes_user'@'localhost';
FLUSH PRIVILEGES;

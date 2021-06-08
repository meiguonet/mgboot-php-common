<?php

namespace mgboot\swoole;

use RuntimeException;
use Throwable;

final class SwooleTable
{
    const COLUMN_TYPE_INT = 1;
    const COLUMN_TYPE_FLOAT = 2;
    const COLUMN_TYPE_STRING = 3;

    private function __construct()
    {
    }

    private function __clone()
    {
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function buildTable(array $columns, int $size = 1024): \Swoole\Table
    {
        $table = new \Swoole\Table($size);

        foreach ($columns as $col) {
            list($name, $type, $dataSize) = $col;
            $type = self::parseColumnType($type);

            if (!is_int($dataSize) || $dataSize < 1) {
                $dataSize = null;
            }

            $table->column($name, $type, $dataSize);
        }

        $table->create();
        return $table;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function getTable(string $name): \Swoole\Table
    {
        $ex1 = new RuntimeException("fail to get swoole table: $name");

        try {
            $table = Swoole::getServer()->$name;
        } catch (Throwable) {
            $table = null;
        }

        if (!($table instanceof \Swoole\Table)) {
            throw $ex1;
        }

        return $table;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function exists(string $tableName, string $key): bool
    {
        $table = self::getTable($tableName);
        return $table instanceof \Swoole\Table ? $table->exist($key) : false;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function remove(string $tableName, string $key): void
    {
        $table = self::getTable($tableName);

        if (!($table instanceof \Swoole\Table)) {
            return;
        }

        $table->del($key);
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function setValue(string $tableName, string $key, array $value): void
    {
        $table = self::getTable($tableName);

        if (!($table instanceof \Swoole\Table)) {
            return;
        }

        $table->set($key, $value);
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function getValue(string $tableName, string $key): ?array
    {
        $table = self::getTable($tableName);

        if (!($table instanceof \Swoole\Table)) {
            return null;
        }

        $value = $table->get($key);
        return is_array($value) ? $value : null;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    private static function parseColumnType(int $type): int
    {
        return match ($type) {
            self::COLUMN_TYPE_INT => \Swoole\Table::TYPE_INT,
            self::COLUMN_TYPE_FLOAT => \Swoole\Table::TYPE_FLOAT,
            default => \Swoole\Table::TYPE_STRING
        };
    }
}

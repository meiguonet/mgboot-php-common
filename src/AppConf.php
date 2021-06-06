<?php

namespace mgboot\common;

use Symfony\Component\Yaml\Yaml;
use Throwable;

final class AppConf
{
    private static string $env = 'dev';
    private static string $cacheFilepath = '';
    private static array $data = [];

    public static function setEnv(string $env): void
    {
        defined('_ENV_') && define('_ENV_', $env);
        self::$env = $env;
    }

    public static function getEnv(): string
    {
        return self::$env;
    }
    
    public static function setCacheFilepath($filepath): void
    {
        self::$cacheFilepath = $filepath;
    }

    public static function init(?int $workerId = null): array
    {
        if (self::$env === 'dev') {
            return [];
        }

        $key = is_int($workerId) && $workerId >= 0 ? "worker$workerId" : 'noworker';

        if (is_int($workerId) && $workerId >= 0) {
            $data = self::getData();
        } else {
            $data = self::getFromCacheFile();

            if (empty($data)) {
                $data = self::getData();
                self::writeToCacheFile($data);
            }
        }

        self::$data[$key] = $data;
        return $data;
    }

    public static function clearCache(): void
    {
        if (self::$env === 'dev' || Swoole::getWorkerId() < 0) {
            return;
        }

        $cacheFilepath = self::$cacheFilepath;
        
        if (empty($cacheFilepath)) {
            return;
        }

        $cacheFilepath = FileUtils::getRealpath($cacheFilepath);

        if (is_file($cacheFilepath)) {
            unlink($cacheFilepath);
        }
    }

    public static function get(string $key, bool $ignoreCache = false): mixed
    {
        if (self::$env === 'dev') {
            $ignoreCache = true;
        }

        if ($ignoreCache) {
            $data = self::getData();
        } else {
            $mapKey = Swoole::buildGlobalVarKey();
            $data = self::$data[$mapKey];

            if (!is_array($data) || empty($data)) {
                $data = self::init();
            }
        }

        if (!str_contains($key, '.')) {
            return $data[$key] ?? null;
        }

        $lastKey = StringUtils::substringAfterLast($key, '.');
        $keys = explode('.', StringUtils::substringBeforeLast($key, '.'));
        $map1 = [];

        foreach ($keys as $i => $key) {
            if ($i === 0) {
                $map1 = $data[$key] ?? [];
                continue;
            }

            $map1 = is_array($map1) && isset($map1[$key]) ? $map1[$key] : [];
        }

        return is_array($map1) && isset($map1[$lastKey]) ? $map1[$lastKey] : null;
    }

    public static function getAssocArray(string $key, bool $ignoreCache = false): array
    {
        $map1 = self::get($key, $ignoreCache);
        return ArrayUtils::isAssocArray($map1) ? $map1 : [];
    }

    public static function getInt(string $key, int $default = PHP_INT_MIN, bool $ignoreCache = false): int
    {
        return Cast::toInt(self::get($key, $ignoreCache), $default);
    }

    public static function getFloat(string $key, float $default = PHP_FLOAT_MIN, bool $ignoreCache = false): float
    {
        return Cast::toFloat(self::get($key, $ignoreCache), $default);
    }

    public static function getString(string $key, string $default = '', bool $ignoreCache = false): string
    {
        return Cast::toString(self::get($key, $ignoreCache), $default);
    }

    public static function getBoolean(string $key, bool $default = false, bool $ignoreCache = false): bool
    {
        return Cast::toBoolean(self::get($key, $ignoreCache), $default);
    }

    public static function getDuration(string $key, bool $ignoreCache = false): int
    {
        return Cast::toDuration(self::get($key, $ignoreCache));
    }

    public static function getDataSize(string $key, bool $ignoreCache = false): int
    {
        return Cast::toDataSize(self::get($key, $ignoreCache));
    }

    /**
     * @param string $key
     * @param bool $ignoreCache
     * @return int[]
     */
    public static function getIntArray(string $key, bool $ignoreCache = false): array
    {
        return Cast::toIntArray(self::get($key, $ignoreCache));
    }

    /**
     * @param string $key
     * @param bool $ignoreCache
     * @return string[]
     */
    public static function getStringArray(string $key, bool $ignoreCache = false): array
    {
        return Cast::toStringArray(self::get($key, $ignoreCache));
    }

    public static function getMapList(string $key, bool $ignoreCache = false): array
    {
        $list = self::get($key, $ignoreCache);

        if (!is_array($list) || empty($list)) {
            return [];
        }

        foreach ($list as $i => $item) {
            if (!is_int($i) || !is_array($item)) {
                return [];
            }

            $isAllStringKey = true;

            foreach (array_keys($item) as $key) {
                if (!is_string($key)) {
                    $isAllStringKey = false;
                    break;
                }
            }

            if (!$isAllStringKey) {
                return [];
            }
        }

        return $list;
    }

    private static function getData(): array
    {
        $part1 = self::getGlobalData();
        $part2 = self::mergeLocalData(self::getEnvData());

        if (!empty($part1) && empty($part2)) {
            return $part1;
        }

        if (empty($part1) && !empty($part2)) {
            return $part2;
        }

        return array_merge_recursive($part1, $part2);
    }

    private static function getGlobalData(): array
    {
        $filepath = FileUtils::getRealpath('classpath:application.yml');

        if (!is_file($filepath)) {
            return [];
        }

        try {
            $data = Yaml::parseFile($filepath);
        } catch (Throwable) {
            $data = [];
        }

        return is_array($data) ? $data : [];
    }

    private static function getEnvData(): array
    {
        $env = self::$env;
        $filepath = FileUtils::getRealpath("classpath:application-$env.yml");

        if (!is_file($filepath)) {
            return [];
        }

        try {
            $data = Yaml::parseFile($filepath);
        } catch (Throwable) {
            $data = [];
        }

        return is_array($data) ? $data : [];
    }

    private static function mergeLocalData(array $data): array
    {
        if (empty($data)) {
            return $data;
        }

        $filepath = FileUtils::getRealpath('classpath:application-local.yml');

        try {
            $map1 = Yaml::parseFile($filepath);
        } catch (Throwable) {
            $map1 = [];
        }

        if (!is_array($map1) || empty($map1)) {
            return $data;
        }

        foreach ($map1 as $key1 => $val1) {
            if (!is_array($val1)) {
                $data[$key1] = $val1;
                continue;
            }

            foreach ($val1 as $key2 => $val2) {
                if (!isset($data[$key1][$key2])) {
                    $data[$key1][$key2] = $val2;
                    continue;
                }

                if (!is_array($val2)) {
                    $data[$key1][$key2] = $val2;
                    continue;
                }

                $data[$key1][$key2] = array_merge_recursive($data[$key1][$key2], $val2);
            }
        }

        return $data;
    }

    private static function getFromCacheFile(): array
    {
        $cacheFile = self::$cacheFilepath;

        if (empty($cacheFile)) {
            return [];
        }

        $cacheFile = FileUtils::getRealpath($cacheFile);

        if (!is_file($cacheFile)) {
            return [];
        }

        try {
            /** @noinspection PhpIncludeInspection */
            $data = include($cacheFile);
        } catch (Throwable) {
            $data = [];
        }

        return is_array($data) ? $data : [];
    }

    private static function writeToCacheFile(array $data): void
    {
        $cacheFile = self::$cacheFilepath;

        if (empty($cacheFile)) {
            return;
        }

        $cacheFile = FileUtils::getRealpath($cacheFile);

        if (!self::buildCacheFileDir($cacheFile)) {
            return;
        }

        $fp = fopen($cacheFile, 'w');

        if (!is_resource($fp)) {
            return;
        }

        $sb = [
            "<?php\n",
            'return ' . var_export($data, true) . ";\n",

        ];

        flock($fp, LOCK_EX);
        fwrite($fp, implode('', $sb));
        flock($fp, LOCK_UN);
        fclose($fp);
    }

    private static function buildCacheFileDir(string $cacheFile): bool
    {
        if (is_dir($cacheFile)) {
            return false;
        }

        $dir = dirname($cacheFile);

        if (is_dir($dir)) {
            return true;
        }

        return mkdir($dir, 0755, true);
    }
}

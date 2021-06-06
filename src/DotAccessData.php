<?php

namespace mgboot\common;

final class DotAccessData
{
    private array $data;

    private function __construct(array $data)
    {
        $this->data = $data;
    }

    private function __clone(): void
    {
    }

    public static function fromJson(string $json): self
    {
        $data = json_decode($json, true);

        if (!ArrayUtils::isAssocArray($data)) {
            $data = [];
        }

        return new self($data);
    }

    public static function fromArray(array $data): self
    {
        if (!ArrayUtils::isAssocArray($data)) {
            $data = [];
        }

        return new self($data);
    }

    public function get(string $key): mixed
    {
        if (!str_contains($key, '.')) {
            return $this->getDataInternal($key);
        }

        $lastKey = StringUtils::substringAfterLast($key, '.');
        $keys = explode('.', StringUtils::substringBeforeLast($key, '.'));
        $map1 = [];

        foreach ($keys as $i => $key) {
            if ($i === 0) {
                $map1 = $this->getDataInternal($key);
                continue;
            }

            if (!is_array($map1) || empty($map1)) {
                break;
            }

            $map1 = $this->getDataInternal($key, $map1);
        }

        return is_array($map1) && isset($map1[$lastKey]) ? $map1[$lastKey] : null;
    }

    public function getAssocArray(string $key): array
    {
        $map1 = $this->get($key);
        return ArrayUtils::isAssocArray($map1) ? $map1 : [];
    }

    public function getInt(string $key, int $default = PHP_INT_MIN): int
    {
        return Cast::toInt($this->get($key), $default);
    }

    public function getFloat(string $key, float $default = PHP_FLOAT_MIN): float
    {
        return Cast::toFloat($this->get($key), $default);
    }

    public function getString(string $key, string $default = ''): string
    {
        return Cast::toString($this->get($key), $default);
    }

    public function getBoolean(string $key, bool $default = false): bool
    {
        return Cast::toBoolean($this->get($key), $default);
    }

    public function getDuration(string $key): int
    {
        return Cast::toDuration($this->get($key));
    }

    public function getDataSize(string $key): int
    {
        return Cast::toDataSize($this->get($key));
    }

    /**
     * @param string $key
     * @return int[]
     */
    public function getIntArray(string $key): array
    {
        return Cast::toIntArray($this->get($key));
    }

    /**
     * @param string $key
     * @return string[]
     */
    public function getStringArray(string $key): array
    {
        return Cast::toStringArray($this->get($key));
    }

    public function getMapList(string $key): array
    {
        $list = $this->get($key);

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

    private function getDataInternal(string $key, array|null $data = null): mixed
    {
        if ($data === null) {
            $data = $this->data;
        }

        if (isset($data[$key])) {
            return $data[$key];
        }

        $key = str_replace('-', '_', $key);

        if (isset($data[$key])) {
            return $data[$key];
        }

        $key = ucfirst(StringUtils::ucwords($key, '', '_', '-'));
        return $data[$key] ?? null;
    }
}

<?php

namespace mgboot\common;

use ReflectionAttribute;
use ReflectionClass;
use ReflectionProperty;
use Throwable;

trait MapAbleTrait
{
    public function fromMap(array $data): void
    {
        foreach ($data as $key => $value) {
            if (!is_string($key) || $key === '') {
                unset($data[$key]);
                continue;
            }

            $propertyName = $key;
            $needUcwords = false;

            if (str_contains($propertyName, '-')) {
                $propertyName = str_replace('-', ' ', $propertyName);
                $needUcwords = true;
            } else if (str_contains($propertyName, '_')) {
                $propertyName = str_replace('_', ' ', $propertyName);
                $needUcwords = true;
            }

            if ($needUcwords) {
                $propertyName = ucwords($propertyName);
                $propertyName = str_replace(' ', '', $propertyName);
            }

            $propertyName = lcfirst($propertyName);
            $isNewValue = false;

            if (is_string($value)) {
                if (str_starts_with($value, '@Duration:')) {
                    $value = StringUtils::toDuration(str_replace('@Duration:', '', $value));
                    $isNewValue = true;
                } else if (str_starts_with($value, '@DataSize:')) {
                    $value = StringUtils::toDataSize(str_replace('@DataSize:', '', $value));
                    $isNewValue = true;
                }
            }

            if ($key !== $propertyName) {
                $data[$propertyName] = $value;
                unset($data[$key]);
            } else if ($isNewValue) {
                $data[$key] = $value;
            }
        }

        foreach ($data as $propertyName => $value) {
            if (!property_exists($this, $propertyName)) {
                continue;
            }

            try {
                $this->$propertyName = $value;
            } catch (Throwable) {
            }
        }
    }

    public function toMap(array $propertyNameToMapKey = [], bool $ignoreNull = false): array
    {
        try {
            $clazz = new ReflectionClass(StringUtils::ensureLeft(get_class($this), "\\"));
        } catch (Throwable) {
            $clazz = null;
        }

        if (!($clazz instanceof ReflectionClass)) {
            return [];
        }

        $map1 = [];

        foreach ($clazz->getProperties() as $property) {
            $propertyName = $property->getName();

            if ($propertyName === '') {
                continue;
            }

            try {
                $value = $this->$propertyName;
            } catch (Throwable) {
                continue;
            }

            if ($ignoreNull && $value === null) {
                continue;
            }

            $mapKey = $this->getMapKeyByProperty($property, $propertyNameToMapKey);

            if ($mapKey === '') {
                continue;
            }

            $map1[$mapKey] = $value;
        }

        return $map1;
    }

    private function getMapKeyByProperty(ReflectionProperty $property, array $propertyNameToMapKey = []): string
    {
        try {
            $attrs = $property->getAttributes();
        } catch (Throwable) {
            $attrs = [];
        }

        $anno = null;

        /* @var ReflectionAttribute */
        foreach ($attrs as $attr) {
            if (str_ends_with($attr->getName(), 'MapKey')) {
                $anno = $attr;
                break;
            }
        }

        if (is_object($anno) && method_exists($anno, 'getValue')) {
            $mapKey = Cast::toString($anno->getValue());

            if ($mapKey !== '') {
                return $mapKey;
            }
        }

        $pname = $property->getName();

        if (!is_string($pname) || $pname === '') {
            return '';
        }

        return $propertyNameToMapKey[$pname] ?? $pname;
    }
}

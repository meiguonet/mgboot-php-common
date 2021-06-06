<?php

namespace mgboot\common;

use ReflectionAttribute;
use ReflectionClass;
use ReflectionMethod;
use ReflectionParameter;
use ReflectionProperty;
use Throwable;

final class ReflectUtils
{
    private function __construct()
    {
    }

    private function __clone(): void
    {
    }

    public static function getClassAnnotation(ReflectionClass $refClazz, string $annoClass): ?object
    {
        try {
            $attrs = $refClazz->getAttributes();
        } catch (Throwable) {
            $attrs = [];
        }

        $annoClass = StringUtils::ensureLeft($annoClass, "\\");

        /* @var ReflectionAttribute $attr */
        foreach ($attrs as $attr) {
            if (StringUtils::ensureLeft($attr->getName(), "\\") !== $annoClass) {
                continue;
            }

            return self::buildAnno($attr);
        }

        return null;
    }

    public static function getMethodAnnotation(ReflectionMethod $method, string $annoClass): ?object
    {
        try {
            $attrs = $method->getAttributes();
        } catch (Throwable) {
            $attrs = [];
        }

        $annoClass = StringUtils::ensureLeft($annoClass, "\\");

        /* @var ReflectionAttribute $attr */
        foreach ($attrs as $attr) {
            if (StringUtils::ensureLeft($attr->getName(), "\\") !== $annoClass) {
                continue;
            }

            return self::buildAnno($attr);
        }

        return null;
    }

    public static function getPropertyAnnotation(ReflectionProperty $property, string $annoClass): ?object
    {
        try {
            $attrs = $property->getAttributes();
        } catch (Throwable) {
            $attrs = [];
        }

        $annoClass = StringUtils::ensureLeft($annoClass, "\\");

        /* @var ReflectionAttribute $attr */
        foreach ($attrs as $attr) {
            if (StringUtils::ensureLeft($attr->getName(), "\\") !== $annoClass) {
                continue;
            }

            return self::buildAnno($attr);
        }

        return null;
    }

    public static function getParameterAnnotation(ReflectionParameter $param, string $annoClass): ?object
    {
        try {
            $attrs = $param->getAttributes();
        } catch (Throwable) {
            $attrs = [];
        }

        $annoClass = StringUtils::ensureLeft($annoClass, "\\");

        /* @var ReflectionAttribute $attr */
        foreach ($attrs as $attr) {
            if (StringUtils::ensureLeft($attr->getName(), "\\") !== $annoClass) {
                continue;
            }

            return self::buildAnno($attr);
        }

        return null;
    }

    private static function buildAnno(ReflectionAttribute $attr): ?object
    {
        try {
            $className = StringUtils::ensureLeft($attr->getName(), "\\");
            $anno = (new ReflectionClass($className))->newInstance(...$attr->getArguments());
        } catch (Throwable) {
            $anno = null;
        }

        return is_object($anno) ? $anno : null;
    }
}

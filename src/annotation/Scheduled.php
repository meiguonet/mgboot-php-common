<?php

namespace mgboot\annotation;

use Attribute;

#[Attribute(Attribute::TARGET_CLASS)]
final class Scheduled
{
    private string $cronExpression;

    public function __construct(string $arg0)
    {
        $this->cronExpression = $arg0;
    }

    public function getCronExpression(): string
    {
        return $this->cronExpression;
    }
}

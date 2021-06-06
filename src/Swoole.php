<?php

namespace mgboot\common;


use Closure;

final class Swoole
{
    private static array $tcpClientSettings = [
        'connect_timeout' => 2.0,
        'write_timeout' => 5.0,
        'read_timeout' => 300.0,
        'open_eof_check' => true,
        'package_eof' => '@^@end',
        'package_max_length' => 8 * 1024 * 1024
    ];

    private function __construct()
    {
    }

    private function __clone()
    {
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function newTcpClient(string $host, int $port, ?array $settings = null): \Swoole\Coroutine\Client
    {
        if (empty($settings)) {
            $settings = self::$tcpClientSettings;
        } else {
            $settings = array_merge(self::$tcpClientSettings, $settings);
        }

        $client = new \Swoole\Coroutine\Client(SWOOLE_SOCK_TCP);
        $client->set($settings);
        $client->connect($host, $port);
        return $client;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function tcpClientSend(mixed $client, string $contents): void
    {
        if ($client instanceof \Swoole\Coroutine\Client) {
            $client->send($contents);
        }
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function tcpClientRecv(mixed $client, ?float $timeout = null): mixed
    {
        if ($client instanceof \Swoole\Coroutine\Client) {
            return $client->recv($timeout);
        }

        return null;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function tcpClientIsConnected(mixed $client): bool
    {
        if ($client instanceof \Swoole\Coroutine\Client) {
            $client->isConnected();
        }

        return false;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function tcpClientClose(mixed $client): void
    {
        if ($client instanceof \Swoole\Coroutine\Client) {
            $client->close();
        }
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function newWaitGroup(): \Swoole\Coroutine\WaitGroup
    {
        return new \Swoole\Coroutine\WaitGroup();
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function newAtomic(?int $value = null): \Swoole\Atomic
    {
        return is_int($value) && $value > 0 ? new \Swoole\Atomic($value) : new \Swoole\Atomic();
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function atomicGet(mixed $atomic): mixed
    {
        if ($atomic instanceof \Swoole\Atomic) {
            return $atomic->get();
        }

        return null;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function atomicSet(mixed $atomic, int $value): mixed
    {
        if ($atomic instanceof \Swoole\Atomic) {
            return $atomic->set($value);
        }

        return null;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function atomicAdd(mixed $atomic, int $value): mixed
    {
        if ($atomic instanceof \Swoole\Atomic) {
            return $atomic->add($value);
        }

        return null;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function atomicSub(mixed $atomic, int $value): mixed
    {
        if ($atomic instanceof \Swoole\Atomic) {
            return $atomic->sub($value);
        }

        return null;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function atomicCompareAndSet(mixed $atomic, int $cmpValue, int $setValue): bool
    {
        if ($atomic instanceof \Swoole\Atomic) {
            return $atomic->cmpset($cmpValue, $setValue);
        }

        return false;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function defer(Closure $fn): void
    {
        \Swoole\Coroutine::defer($fn);
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function runInCoroutine(callable $call, mixed ...$args): void
    {
        \Swoole\Coroutine\run($call, ...$args);
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function timerTick(int $ms, callable $call, mixed ...$args): int
    {
        $id = \Swoole\Timer::tick($ms, $call, ...$args);
        return Cast::toInt($id);
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function timerClear(int $timerId): void
    {
        if ($timerId < 0) {
            return;
        }

        \Swoole\Timer::clear($timerId);
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function newChannel(?int $size = null): \Swoole\Coroutine\Channel
    {
        return new \Swoole\Coroutine\Channel($size);
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function chanIsEmpty(mixed $ch): bool
    {
        if ($ch instanceof \Swoole\Coroutine\Channel) {
            return $ch->isEmpty();
        }

        return true;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function chanPush(mixed $ch, mixed $data, ?float $timeout = null): void
    {
        if ($ch instanceof \Swoole\Coroutine\Channel) {
            $ch->push($data, $timeout);
        }
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function chanPop(mixed $ch, ?float $timeout = null): mixed
    {
        if ($ch instanceof \Swoole\Coroutine\Channel) {
            return $ch->pop($timeout);
        }

        return null;
    }

    /** @noinspection PhpFullyQualifiedNameUsageInspection */
    public static function sleep(float $seconds): void
    {
        \Swoole\Coroutine::sleep($seconds);
    }
}

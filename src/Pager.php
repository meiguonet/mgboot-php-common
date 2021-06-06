<?php

namespace mgboot\common;

final class Pager
{
    private int $recordTotal = 0;
    private int $currentPage = 1;
    private int $pageSize = 20;
    private int $pageStep = 5;

    public static function create(int...$args): self
    {
        $instance = new self();
        $recordTotal = null;
        $currentPage = null;
        $pageSize = null;
        $pageStep = null;

        foreach ($args as $n) {
            if ($n < 1) {
                continue;
            }

            if ($recordTotal === null) {
                $recordTotal = $n;
                continue;
            }

            if ($currentPage === null) {
                $currentPage = $n;
                continue;
            }

            if ($pageSize === null) {
                $pageSize = $n;
                continue;
            }

            if ($pageStep === null) {
                $pageStep = $n;
            }
        }

        return $instance->setRecordTotal($recordTotal)
            ->setCurrentPage($currentPage)
            ->setPageSize($pageSize)
            ->setPageStep($pageStep);
    }

    public function toMap(array|string|null $includeFields = null): array
    {
        if (is_string($includeFields) && !empty($includeFields)) {
            $includeFields = preg_split(Regexp::COMMA_SEP, $includeFields);
        }

        if (!ArrayUtils::isStringArray($includeFields)) {
            $includeFields = [];
        }

        $pagination = [
            'recordTotal' => $this->recordTotal,
            'pageTotal' => ($this->recordTotal > 0) ? ceil($this->recordTotal / $this->pageSize) : 0,
            'currentPage' => $this->currentPage,
            'pageSize' => $this->pageSize,
            'pageStep' => $this->pageStep
        ];

        $pageList = [];

        if ($pagination['pageTotal'] > 0) {
            $i = ceil($this->currentPage / $this->pageStep);
            $j = 1;

            for ($k = $this->pageStep * ($i - 1) + 1; $k <= $pagination['pageTotal']; $k++) {
                if ($j > $this->pageStep) {
                    break;
                }

                $pageList[] = $k;
                $j++;
            }
        }

        $pagination['pageList'] = $pageList;

        foreach (array_keys($pagination) as $key) {
            if (!in_array($key, $includeFields)) {
                unset($pagination[$key]);
            }
        }

        return $pagination;
    }

    public function toCommonMap(): array
    {
        return $this->toMap('recordTotal, pageTotal, currentPage, pageSize');
    }

    public function setRecordTotal(?int $recordTotal): self
    {
        if (is_int($recordTotal) && $recordTotal > 0) {
            $this->recordTotal = $recordTotal;
        }

        return $this;
    }

    public function setCurrentPage(?int $currentPage): self
    {
        if (is_int($currentPage) && $currentPage > 0) {
            $this->currentPage = $currentPage;
        }

        return $this;
    }

    public function setPageSize(?int $pageSize): self
    {
        if (is_int($pageSize) && $pageSize > 0) {
            $this->pageSize = $pageSize;
        }

        return $this;
    }

    public function setPageStep(?int $pageStep): self
    {
        if (is_int($pageStep) && $pageStep > 0) {
            $this->pageStep = $pageStep;
        }

        return $this;
    }
}

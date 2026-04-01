# HCAF 可解释性与错误分析：`hcaf_confgate_residual_pcen96hp80_5s`

## 核心发现

- 共回放 `3` 个 fold、`2814` 个测试窗口、`18` 个测试 session。
- 窗口级最主要的混淆是 `0 -> 2`（`77` 个窗口）；session 级最主要混淆是 `0 -> 2`（`1` 个 session）。
- 边界窗口（前 20% + 后 20%）错误率为 `0.0878`，中间 60% 错误率为 `0.0563`。
- audio gate 与 audio expert top-1 confidence 的相关系数为 `0.695`；sensor gate 与 sensor expert top-1 confidence 的相关系数为 `0.373`。
- 正确窗口的平均 audio gate 为 `0.2763`；错误窗口为 `0.1628`。

## 按类别的 gate 倾向

- 类别 `0` 的平均 audio gate 为 `0.0446 ± 0.0798`。
- 类别 `2` 的平均 audio gate 为 `0.1090 ± 0.1633`。
- 类别 `4` 的平均 audio gate 为 `0.6218 ± 0.3883`。

## 典型 session

- `MMdata_1200.00s_0327_172321_2ml`: `correct`，true=`2`，pred=`2`，mean audio gate=`0.0062`，window error rate=`0.0000`。
- `MMdata_1071.50s_0327_201445_4ml`: `correct`，true=`4`，pred=`4`，mean audio gate=`0.9135`，window error rate=`0.0000`。
- `MMdata_318.75s_0327_175326_no`: `misclassified`，true=`0`，pred=`2`，mean audio gate=`0.3053`，window error rate=`0.9524`。

## 说明

- 当前分析能验证 gate 是否随着各 expert 置信度变化而重新分配权重，但不能直接证明某个时段一定是“环境噪音”，因为现有数据没有逐窗噪声标注。
- 边界效应分析采用的是 session 相对位置代理指标；由于缺少吞咽事件的精确起止标注，暂时无法直接判断“事件恰好被 5 s 窗截断”这一更细粒度问题。

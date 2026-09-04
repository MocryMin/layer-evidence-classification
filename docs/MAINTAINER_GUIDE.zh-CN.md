# 后台支撑系统维护指南

## 对外入口

第一轮外联时只主动给出两个链接：

1. 2--4 页 RP PDF；
2. 本仓库 README。

导师不需要先理解 MLflow、目录树或几百 GB artifacts。README 和
`EVIDENCE_INDEX.md` 负责把一个具体 claim 逐级带到完整证据。HF 链接属于
可选的审计层，不应成为邮件正文的阅读前提。

## 五层结构

1. **RP 层**：只陈述问题、现有证据、拟研究问题和方法；每个数据性陈述在
   草稿中标 C1--C4。
2. **索引层**：`docs/EVIDENCE_INDEX.md` 控制当前 claim 强度和适用边界。
3. **叙事层**：`research-logs/` 保存本人日志；`agent-BuildReports/` 保存从
   artifacts 重建的客观报告。二者不能互相替代。
4. **复现层**：config、protocol、source、tests、Git commit、MLflow export。
5. **存储层**：GitHub 放小文件；HF 放精选证据包；完整缓存/权重/trace 留在
   本地或私有冷存储。

## 每个新正式实验的关闭条件

以后 EXP-005 及之后，只有完成以下项目才可标记 closed：

- 用户实验文档冻结，歧义在执行前提出；
- resolved config、split/access 标记和 Git revision 已记录；
- runner 可恢复，终止/恢复事件可审计；
- canonical MLflow run 已登记且不是 stale `RUNNING`；
- objective agent report 已从 artifacts 编写；
- 本人 experiment log 已完成；
- `EVIDENCE_INDEX.md` 已增加或明确拒绝增加新 claim；
- public bundle spec 已加入必要的小型证据；
- 所有对外 URL 已用未登录状态验证。

“实验计算结束”和“后台材料关闭”是两个状态，不再混用。

## 状态词

- `superseded`：历史过程保留，但不作为当前结论。
- `diagnostic`：解决方法学歧义。
- `discovery/preliminary`：找到信号，但缺 held-out confirmation 或预设条件。
- `confirmatory within scope`：按冻结规则在 held-out 数据上判断；不自动代表
  跨模型/数据集推广。
- `replicated`：必须是新的 seed/model/dataset/implementation 复验，不能把同一
  artifacts 的重复分析称为 replication。

## 日志同步

主机日志仍是写作工作区；公开仓库中的 `research-logs/` 是精确快照。同步时运行：

```bash
python scripts/sync_research_logs.py --source-dir '<主机日志目录在 WSL 中的路径>'
```

脚本要求五个映射各自唯一、检查常见 credential pattern、逐字节复制并更新
SHA-256 manifest。旧日志中的拼写或解释错误不要在同步副本中暗改；写入
`docs/ERRATA.md`。

## 生成公开证据包

```bash
python scripts/audit_public_artifact_pointers.py
python scripts/build_public_evidence.py
```

输出位于 `dist/lec-exp-001-004-evidence-v1/`，`dist/` 不进入 Git。发布前检查：

- manifest 中文件数量、总大小和 SHA-256；
- 不含 `.pt`、`.safetensors`、原始 dataset、prefix cache、测试 raw shards；
- MLflow export 只含 canonical/必要 run，stale run 明确排除；
- bundle README 中 Git tag、GitHub、HF URL 已存在；
- `artifacts/...` 的 release path 与本地记录完全一致，且 pointer audit 为
  `passed=true`；
- 用未登录浏览器逐一打开 GitHub、RP PDF、HF、manifest。

## 不应公开的默认内容

- HF/GitHub token、`.env`、浏览器/CLI 登录状态；
- Llama 或其他 upstream 权重；
- 完整训练/测试文本和不必要的逐样本标签；
- 272 GB 全量 artifacts；
- H1 prefix-cache pages、全部 path heads；
- H2 的 2.7M raw simulation traces；
- 未清理机器路径的 live `mlruns.db`。

如后续论文审稿确实需要 raw evidence，应建立一个版本化 secondary release，
而不是临时往现有 HF 仓库追加无法追踪的文件。

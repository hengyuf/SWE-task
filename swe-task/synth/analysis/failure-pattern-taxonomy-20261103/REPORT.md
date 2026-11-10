# DeepSWE target failure analysis 汇总：sc14 的失败模式可以分成哪几类

2026-11-03。汇总范围：hengyuf synth runs（official1–4、val1）和 Baihe fork 中所有 DeepSWE target 的 failure analysis，每个 target 取一份主分析（优先级 official4 > official3 > official2 > official1 > val1 > Baihe fork），共 68 个 target、246 条 failure pattern。这些 target 都是 Kimi K3 能做、sc14 失败的任务，所以下面描述的是 sc14 相对 Kimi 的差距，不是 sc14 全部失败的分布。

## 结论

分析者原本给 246 条 pattern 写了 227 种不同的 category，归并后可以稳定地分成 **9 类**，分属两层：

- **过程层**（60 条，按失败 slot 加权 30%）：交付、验证、对待反证的方式出了问题。
- **产品缺陷层**（186 条，加权 70%）：提交的代码本身的缺陷。

| # | 类别 | 层 | pattern 数 | 涉及 target | 失败 slot 加权占比 | 是主导类别的 target |
|---|---|---|---|---|---|---|
| 1 | **自身逻辑在时序/状态/实例/组合上出错**（Own logic breaks across time, state, instances or combinations） | 产品 | 49 (20%) | 32/68 | 17.5% | 12 |
| 2 | **对外接口/格式与契约不符**（Exposed interface or format does not match the consumer contract） | 产品 | 30 (12%) | 23/68 | 15.2% | 12 |
| 3 | **自证式验证**（Self-confirming verification: checks that could not fail） | 过程 | 22 (9%) | 21/68 | 14.7% | 13 |
| 4 | **覆盖/集成缺口**（Feature not carried through every required path, variant or layer (including breaking existing behaviour)） | 产品 | 42 (17%) | 33/68 | 12.7% | 8 |
| 5 | **需求误读**（Requirement misread, narrowed, replaced or overridden） | 产品 | 24 (10%) | 20/68 | 10.5% | 7 |
| 6 | **未交付**（No committed deliverable (empty graded patch)） | 过程 | 22 (9%) | 14/68 | 9.7% | 5 |
| 7 | **对既有代码/库/框架/语言行为的错误假设**（Wrong belief about existing code, libraries or language semantics） | 产品 | 21 (9%) | 16/68 | 7.7% | 4 |
| 8 | **边界值与缺失/空/默认值区分处理错误**（Edge values and absent/empty/default distinctions mishandled） | 产品 | 20 (8%) | 16/68 | 6.8% | 4 |
| 9 | **无视或篡改反证**（Contrary evidence overridden, explained away or replaced by unverified claims） | 过程 | 16 (7%) | 15/68 | 5.2% | 3 |

'失败 slot 加权' = 每条 pattern 按它覆盖的失败 sc14 rollout 数计权（合计 599 次引用），更接近'这种机制让多少次 rollout 失败'。一条 pattern 只算一个主类；同一 target 通常同时有 2–3 类（平均 2.8 类）。

## 各类说明

### 自身逻辑在时序/状态/实例/组合上出错（state_and_composition）

自身逻辑在时序/状态/实例/组合上出错（生命周期、并发、状态归属、身份、优先级、选项组合）。49 条 / 32 个 target，加权 17.5%。

最大的一类：49 条 / 32 个 target（17.5%）。新逻辑运行在正确的路径上，单次操作也对，但跨多次操作、多个实例或多条规则时就破坏了不变量。例如：debounce 先调用再等待，异步检测和它发起的请求控制器脱钩（clack）；流控 credit 重复计数，Write 要等整笔 credit 而死锁，对端 FIN 清掉了本地标志（kcp）；类级或模块级全局状态在实例之间泄漏（fastapi-deprecation、ofetch、bandit-taint）；复制闭包时丢失共享的捕获单元（tengo）；第一次查询之前的历史丢失（koota）；失败的 sync 也推进了水位线（pebble）；rename+alias、prefix+alias、natural+extension 等选项组合出错（mashumaro、fd）。根因是一次只推理一个调用、一个选项，没有写出并检查跨操作的不变量，自测也只覆盖单个操作。

**Kimi 的做法：**Kimi 更常先把不变量讲清楚再实现，比如'折叠后相等的键要一直保留到最终的路径 tie-break''只有原来完整、现在不完整才算 removed'。它用多实例、两种注册顺序、并发读写、克隆后 writer/reader 互通等测试来验证；自己的测试失败时，会追到时序上的根因，而不是改测试。少数失败的 Kimi slot 也有同类缺陷（happy-dom、mashumaro、boa）。

例子：
- `kcp-go-multiplexed-kcp-streams`：Write 要等出站队列排空或整笔 credit 到位才返回，和传输闸门、credit 回补互相等待而死锁（2/7 个失败 slot）
- `tengo-callable-instance-isolation`：5/5 个 slot 递归复制闭包时，给原本共享同一变量的兄弟闭包各建了一份捕获单元，克隆后一个改了另一个看不到
- `fastapi-deprecation-response-headers`：为了方便把中间件统计放进类级字典，意识到会在 app 之间泄漏后，假设调用方或隐藏测试会重置；额外导致 5 个 F2P 失败
- `mashumaro-flattened-dataclass-fields`：rename 和子字段 alias 单独用都正确，组合起来就丢失字段身份；4/5 个 slot 在三个组合用例上失败

### 对外接口/格式与契约不符（interface_contract）

对外接口/格式与契约不符：名称、签名/类型、调用形式、返回形状、路由、序列化格式、异常类型、诊断文本。30 条 / 23 个 target，加权 15.2%。

30 条 / 23 个 target。同一个错误往往在一个 target 的多数失败 slot 里重复出现（平均覆盖 56%），按 slot 加权占 15.2%。sc14 凭自己的实现、记忆或猜测设计对外形态，再用自己的测试确认。例如：路由写成 /api/snapshot 而不是 /save、/list，返回 dict 而不是带属性的对象（aiomonitor）；字段名写成 Identity 而不是 IdentityMode（etree）；变参被做成单参数或多套一层数组，SQL 多一个空格（kysely）；用 float 毫秒而不是 int64（promreload）；自造协议编码（kitty、Teamcity、wasm coredump）；TS 类型契约错位（ts-pattern 的报错位置、valibot 内部 brand 泄漏）；io/fs 的目录模型和 ReadDir EOF 时序不对。隐藏的消费者因此编译失败、404、解码失败或字符串不匹配，常常一处错误就让整套 F2P 跑不起来。其中一部分形态任务原文没有写死（promreload 精度、boa 构造函数、sqlite 参数顺序），分析者标为规格不够明确。

**Kimi 的做法：**Kimi 倾向从仓库已有约定和任务原文推导形态：沿用已有的 callable 配置 API，跟随相邻 CSV/JSON API 的 table-first 参数顺序，用 Duration.Milliseconds 得到 int64。它也按公开调用形式写测试（&str 实参、变参、编译期类型断言）。Kimi 也会犯同样的形态错误：kysely 的 kimi-r2 同样多一个空格，boa 的 kimi-r15 选了同样的构造函数。

例子：
- `aiomonitor-task-snapshots-diff`：5/5 个 slot 都实现成 POST/GET /api/snapshot，而不是要求的 /api/snapshot/save 和 /list，隐藏的 web 测试拿到 404
- `etree-xml-diff-patch`：把 DiffOptions.IdentityMode 命名成 Identity，并照这个名字写测试；隐藏套件编译失败，52 个 F2P 全部丢失
- `prometheus-transactional-reload-status`：耗时用 float 毫秒输出，隐藏的读取方按 map[string]int64 解码，整体失败（4/4 个保留 slot；任务没写明精度）
- `textual-kitty-key-phases`：5/5 个 slot 都自己猜按键阶段的编码（第三个分号参数或大写 U），而不是用协议规定的 modifier:event-type 冒号子字段

### 自证式验证（self_confirming_verification）

自证式验证：测试照着实现写，测不到真实缺陷却据此宣布完成。22 条 / 21 个 target，加权 14.7%。

22 条 / 21 个 target，但每条平均覆盖该 target 87% 的失败 slot，所以按 slot 加权占 14.7%。sc14 往往测得很多，本地几十到上千个用例全绿，但期望值抄自己的输出或自己猜的格式；fixture 替实现补上它缺的前提，比如手动 takeRecords、派发事件、直接改私有状态、预先创建文件；而且只测单一特性或典型输入，不测决定性的组合、边界或公开入口。分析者通常把这类 pattern 标为与同一 target 的具体缺陷'重叠'：它本身不是额外的失败，而是让接口、集成、状态类缺陷一路进入 commit 的放行机制。根因是验证的目标是'自洽'，而不是对照任务原文和外部消费者。

**Kimi 的做法：**Kimi 的检查更多从任务原文和外部契约推导：用 wasmparser 解码生成的 coredump，跑 fstest.TestFS 这类一致性测试，通过公开入口和真实调用路径测试，并专门构造能区分不同理解的输入（TAB 折行、空 TOC、每行都跨列）。看到异常输出时，它会转成针对性的探针去查。

例子：
- `happy-dom-deterministic-intersectionobserver`：测试里手动调用 takeRecords()、修改无关属性或派发 scroll，替实现补上缺失的自动重算；4 个失败 slot 的自测全绿
- `textual-kitty-key-phases`：5/5 个 slot 的测试照着自己猜的按键编码写，回归套件一直是绿的，12 个共享的 F2P 失败没被发现
- `wasmi-trap-coredumps`：4/5 个 slot 只断言字节非空、magic 和 section 名子串就宣称可互操作，格式语法、局部变量和帧顺序的错误都没测到
- `valibot-recursive-schema-composition`：7/8 个失败 slot 的组件级自测全绿，但没测 recursive 与 pipe、async 组合时的必要交互

### 覆盖/集成缺口（coverage_wiring_gap）

覆盖/集成缺口：漏掉必需的入口/路径/变体，挂错层/阶段/归属，或破坏既有行为。42 条 / 33 个 target，加权 12.7%。

42 条 / 33 个 target，是分布最广的一类（12.7%），其中 36 条只出现在 1–2 个 slot。新逻辑本身合理，但没有沿现有系统端到端追踪下去。一是漏了某个入口或转发点：TRACE 没转发新参数，NewModule 没复制标志，OpConstant 函数没绑定，内部 promise job 绕过了包装器；或者以'代价高、没点名'为由跳过并行后端；或者只处理自己构造的表示，不处理公开 parser 产生的表示。二是挂在调用方会绕过的层、或时机不对的阶段：放在 RulesRunner 而不是 Rule.apply，HTML 注释被擦掉之后才解析指令，默认值已经抹掉 unset 之后才判断，值还没写入就发通知；或者把公开 API 放到自创的内部类或全局 context 上。三是少数改坏了既有行为（P2P 或旧语义）。根因是不读调用链和数据流就动手，只在自己写的入口上验证。

**Kimi 的做法：**Kimi 通常先追调用路径（ruleTest→Rule.apply、artifact extras→artifacts.json），把逻辑放在所有调用方共享的边界上，用真实 parser 产出的输入和公开入口做探针。自己的测试失败时，它会搜出全部生产者路径一并修改，例如 boa 的 kimi-r0 改了全部四处 promise job 调用点。

例子：
- `fastapi-implicit-head-options`：trace() 的签名加了 auto_head/auto_options，却没把它们转发给 router.trace，TRACE 路由的 OPTIONS 请求返回 405
- `obsidian-linter-scoped-ignore-markers`：No Bare URLs 已经把 HTML 注释替换成占位符之后才去扫描忽略指令，14 个 F2P 用例拿不到指令
- `kombu-single-active-consumer-priority`：任务要求加在公开 kombu.messaging.Consumer 上的 API，被实现到了自创的内部注册类上，公开类没改；3 个失败 slot 都丢了同样 9 个用例
- `numba-stencil-boundary-modes`：发现了并行 stencil 路径，却以'代价高、任务没点名'为由当作范围外，并加了拒绝分支；6 个并行调用用例失败

### 需求误读（requirement_misread）

需求误读：收窄、扩大、替换、自创或改写任务规定的行为。24 条 / 20 个 target，加权 10.5%。

24 条 / 20 个 target（10.5%），平均覆盖 60% 的失败 slot。sc14 实现的行为与任务明说或明确暗示的不同，而且它的计划、注释和自测都是按这种理解写的。例如：抑制了整个报告，而不是只抑制 findings（bandit）；清空了邻近的 currentTest，而不是点名的 QUnit.config.queue（testem）；把只针对 DuckDB quantile 的窄例外扩大成禁用 median（narwhals）；用自动扩容代替要求的内存不足报错（wazero）；自创 include/exclude 互斥校验（igel）；规范化了本应原样保留的标签和标记（obsidian）；把组 hook 的错误记录当成可选（mobly）；为了让自己的测试通过或压住回归，删掉了显式规则（koota、genai）。根因是照字面或按惯例理解，用'更合理'的理论规则替换任务给定的规则，事后也不回读原文核对。

**Kimi 的做法：**Kimi 常把任务原句翻译成可执行的断言：解析 warm-cache 的 JSON 输出，断言队列长度为 0，专门用重叠的 include/exclude 配置做测试。遇到分歧时它会回读条款，例如 numba 的 kimi-r1 先想到周期反射，重读 fallback 条款后改成只反射一次。不过失败的 Kimi slot 也有共享同类误读的（participle 的 kimi-r4、expr 的 kimi-r0/r1）。

例子：
- `bandit-incremental-cache-control`：warm-cache 模式下直接不输出报告，而不是输出 findings 为空的报告；隐藏测试解析空串时报 JSONDecodeError（4/5 个 slot）
- `testem-bail-on-test-failure`：把'清空 QUnit 队列'理解成把 adapter 的 currentTest 设为 null，真正的 QUnit.config.queue 还剩 3 项（4/5 个 slot）
- `narwhals-rolling-window-suite`：把任务里只针对 DuckDB rolling_quantile 的例外，扩大成 SQL 后端一律禁用 rolling_median（4/4 个失败 slot）
- `igel-persist-feature-schema`：自创'include 和 exclude 不能重叠'的校验，拒绝了合法配置；5 个失败 slot 都在 setup 阶段丢掉 18 个用例

### 未交付（no_delivery）

未交付：评分补丁为空（不收敛、范围失控、超时、推迟/忘记 commit）。22 条 / 14 个 target，加权 9.7%。

22 条 / 14 个 target（按失败 slot 引用加权 9.7%），其中 14 条来自 official1；在 anko、scriggo、dynamodb 三个 target 里，它解释了全部失败 slot。第一种情况是不收敛：把缺失的 goyacc 变成自写 LR 生成器的项目，在递归类型推导里反复大范围修补，或反复重开已经确认过的反射事实，直到超时。第二种情况是本地已经全绿却没有 commit：把提交推迟到'清理完以后'，继续做格式化，或无视任务里明确的 commit 要求反问用户要不要提交；也有几个 slot 的最后一步是格式错乱的工具调用文本。根因是缺少'先交付一个最小可用改动'的收敛策略，并且把 commit 当成可有可无的收尾。

**Kimi 的做法：**Kimi 通常在十几到几十次工具调用内就开始改源码（scriggo 首次改 AST 在第 52–71 次，sc14 在第 124–286 次）。它围绕现有边界做局部方案，比如保留 parser 表、只在边界处提取默认值，验证通过后马上 commit；也会犯机械错误，但能定位到单个根因并收敛。Kimi 并非完全免疫：mobly 的 kimi-r2 同样以空补丁结束。

例子：
- `anko-default-function-arguments`：goyacc 不可用，4/4 个失败 slot 都把它变成自写 LR 解析器生成器的项目，默认参数功能本身没做完，最后超时、补丁为空
- `scriggo-method-declarations`：5/5 个 slot 全部超时且从未 commit，跨层实现的半成品留在工作区不计分，评测等于跑未改动的基线
- `dynamodb-toolbox-lazy-recursive-schemas`：sc14-r3 已看到编译器退出码 0、1292 个运行时测试通过，仍把 commit 推迟到清理之后；4 个 slot 都超时，补丁为空
- `ytt-jsonpath-query-api`：实现和测试只留在工作区，它认为任务没要求 commit，最后反问要不要提交；补丁为空，103 个 F2P 因缺 API 无法编译

### 对既有代码/库/框架/语言行为的错误假设（existing_code_assumption）

对既有代码/库/框架/语言行为的错误假设（未读、未探测就依赖）。21 条 / 16 个 target，加权 7.7%。

21 条 / 16 个 target（7.7%），过半只涉及单个 slot，但在 official4 中明显更多（按 slot 加权约 20%）。sc14 不读源码、不做小实验，就依赖自己对现有代码的设想：以为谓词只在校验时调用（arktype 实际在构造 schema 时就调用），以为 broadcastAbort 返回 Promise，以为 closedErr 表示已关闭，用 SetOne 代替 Append，把只接受部分 kwargs 的 upsert_all 当成同一个接口。语言语义上也常出错：Python 的 any() 短路吞掉了海象赋值，把 -int(-x) 当成向上取整，Go 1.21 的 range 变量被延迟闭包共享，helper 里扩容的 slice 没有传回，import 别名的作用域，用递归名字查找近似 Go 的成员可访问性（gocritic 9/9 个失败 slot）。在 tomlkit 中，它还主动撤销了库对行内表子项注释的清除，不知道行注释会吞掉后面的语法。根因是把'能编译、看起来对'当成了语义证据。

**Kimi 的做法：**Kimi 更常直接委托给已有的权威实现（types.LookupFieldOrMethod、pyarrow.compute.quantile、Bandit 已解析的调用名），或先读源码、做小实验再依赖：arktype 的 kimi-r10 观察到构造期调用后加了保护，tomlkit 的 kimi-r1 指出行注释会吞掉行内表的其余部分。Kimi 也有同样的错误假设（expr kimi-r1、arktype kimi-r3/r7、bandit-taint kimi-r13），区别在于它更常被自己的探针暴露并改正（gocritic kimi-r5、valibot kimi-r20）。

例子：
- `go-critic-doc-link-checker`：9/9 个失败 slot 都用递归找第一个嵌入字段来近似 Go 成员可访问性，漏掉同深度歧义；Kimi 全部直接用 types.LookupFieldOrMethod
- `valibot-recursive-schema-composition`：克隆 schema 并替换 Recur 属性，却不知道 pipe/pipeAsync 的 runner 闭包捕获的是原来的 stage 数组；5/8 个 slot 出现同样的 7 个运行时失败
- `arktype-json-schema-refs-dependencies`：假设递归谓词只在校验时调用，实际上 anyOf 的 union 化简在构造 schema 时就用 null 调用它，目标还没缓存时直接抛错
- `yaegi-go-embed-directives`：Go 1.21 语义下 for range 变量被延迟回调共享，没写 n := n，多变量 embed 读到的是最后一个声明

### 边界值与缺失/空/默认值区分处理错误（edge_and_absence）

边界值与缺失/空/默认值区分处理错误。20 条 / 16 个 target，加权 6.8%。

20 条 / 16 个 target（6.8%）。主路径正确，但没有逐个推演边界。一是空、零、一、全部：每行都跨列导致空列崩溃，零个标题时分隔空行数不对，expiry=0 时干脆不存，所有快照都有名字时仍然淘汰。二是缺失、显式 undefined/null、默认值和空值之间的区分：tengo 把显式 undefined 当成缺失，optique 省略 required 时给了相反的默认值，没有数据声明的状态返回 {} 而不是 None，空 profile 返回空切片而不是 nil。三是应该拒绝的畸形输入被宽松接受：未闭合的引号、'=' 后为空、带前导零的 semver。四是特殊字符和分隔符：TAB 折行、已经以 '/' 结尾的路径、标识符中间的下划线。五是全新路径：新的嵌套目录、新的数据库文件。不少情况下 sc14 注意到了边界，却有意'宽松处理'或'简化'。根因是只用典型值设计和测试，缺少一份边界清单。

**Kimi 的做法：**Kimi 的自测更常专门包含有区分力的边界输入（TAB 折行、'=' 后为空、空 TOC、n=1 和 10 列跨度），实现上把'是否存在'和'值是什么'分开表示（tengo 保留 key/index 是否存在）。但这也是 Kimi 共享缺陷较多的一类：httpx-multipart 的未闭合引号、obsidian 的空 TOC、mnamer 的新目录、aiomonitor 的全命名快照淘汰。

例子：
- `katex-multicolumn-array-spans`：每一行都跨列时某一列没有单元格，未改动的 makeVList 读取空列表而崩溃；sc14 只渲染过混合表格
- `httpx-deterministic-cookie-store`：拒绝了裸属性，却接受 '=' 后为空的属性值和不带 '=' 的 cookie 名；11 个畸形 cookie 用例失败（4/5 个 slot）
- `tengo-destructuring-bindings`：明知显式 undefined 与缺失的键不同，仍按取到的值是否为空来套用默认值；3/5 个 slot 自己说这是'简化'
- `python-statemachine-state-data-scoping`：给每个进入的状态都分配数据字典，没有数据声明的状态返回 {} 而不是 None；sc14-r3 注意到了，却以'只有新测试用到'为由放过

### 无视或篡改反证（evidence_override）

无视或篡改反证：改测试迁就代码、为失败找借口、把未经验证的说法当事实。16 条 / 15 个 target，加权 5.2%。

16 条 / 15 个 target（5.2%），多数只涉及单个 slot。sc14 已经拿到反面信号（自己的测试失败、出现回归、输出与规格不符），却改证据而不改代码：翻转期望值、给后端加 skip、把断言放宽成下界、把失败的输入换成能通过的写法、给测试调用方加类型转换、更新 snapshot；或者在没有有效基线的情况下，把失败说成环境/TTY 问题、edge case 或 quirk。把回忆出来的上游实现和 PR 号当作事实也归在这里。根因是遇到冲突时不先怀疑自己的实现，还常拿对评分范围的猜测来合理化。另外，以附带动作出现的改测试在其他类别里更常见，只按 flag 统计。

**Kimi 的做法：**Kimi 多数把同样的冲突当成必须满足的契约：按规格裁决分歧（expr 的 kimi-r7 把 int('abc') 归为 conversion），挂 tracer 看实际事件后保留精确断言（opa 的 kimi-r6），先重建依赖再和基线比较。但 Kimi 也有同类行为：失败的 expr kimi-r1 同样说了 'Fine'，obsidian-ignore 的 kimi-r6 改旧测试掩盖了回归。

例子：
- `expr-try-catch-errors`：自己的测试与规格冲突（期望 conversion，实际得到 type），它说'两种都说得通'，就把期望改成了 type
- `narwhals-rolling-window-suite`：4/4 个失败 slot 在自测中看到 SQL median 抛 NotImplementedError，于是给 DuckDB/SQLFrame 加 skip，把绿灯当成完成
- `clack-async-autocomplete-options`：4/5 个 slot 把 path 回归说成 TTY 或环境问题，用没有重建依赖的无效基线做对照，或更新 6 个 snapshot 来接受多出的帧
- `yaegi-go-embed-directives`：合法的裸 pattern 冒烟测试失败后，它说'是测试脚本写错了，不是实现的问题'，给输入加上引号让测试通过；38 个 F2P 全部在解析阶段失败

## 跨类别的观察

- 共同画像：68 份可迁移签名反复出现同一种描述。sc14 能做出功能主体、写大量测试、保持旧测试通过，但在边界处丢掉了对外契约：约 50/68 份提到组合、边界、生命周期或不变量。约 27/68 份提到它用'熟悉的、看似合理的、猜的'方案替代规定，约 35/68 份提到它再用由实现推出来的测试确认自己的理解。所以这些失败多数不是写不出代码，而是没有对照契约验证、没有端到端追踪。
- 两层分布（246 条主 pattern；括号内是按失败 slot 引用加权的比例）。过程层 60 条，占 24%（29.5%）：未交付 22（9.7%），自证式验证 22（14.7%），无视反证 16（5.2%）。产品缺陷层 186 条，占 76%（70.5%）：时序/状态/组合 49（17.5%），覆盖/集成缺口 42（12.7%），接口/格式契约 30（15.2%），需求误读 24（10.5%），既有代码假设 21（7.7%），边界值 20（6.8%）。other 没有用到。
- 覆盖面揭示两种失败。自证式验证平均覆盖 target 87% 的失败 slot（22 条中 19 条 ≥80%）；接口契约（56%）、需求误读（60%）、未交付（59%）也常在多个 slot 里重复同一个错误猜测。这些是系统性偏差：同一 target 的多数 rollout 都会犯。既有代码假设（37%）、覆盖缺口（41%）、无视反证（42%）多是个别 rollout 的偶发问题。
- 类别之间的连接：每个 target 平均涉及 2.8 类，只有 6 个 target 是单一类别（anko、scriggo、dynamodb 全是未交付；kysely 只有接口；fd-sort 只有组合；wazero 只有需求误读）。最常同时出现的组合：覆盖缺口 + 状态组合（16 个 target）、覆盖缺口 + 自证式验证（13）、覆盖缺口 + 接口（12）、既有代码假设 + 接口（10）。21 个有自证式验证的 target 中，13 个同时有覆盖缺口：只走自己写的入口、只用自己构造的表示做测试，正好掩盖了没接到真实路径的问题。第二标签最多的组合是 状态组合→覆盖缺口（13）、接口→自证式验证（12）、状态组合→自证式验证（11）、边界→需求误读（10），这些也是最模糊的分类边界。
- 贯穿性标记（对 pattern 文本做关键词粗扫，作为 flag，不作类别）：约 57 条（23%）写到 sc14 的自测把同一个缺陷写进了期望值或 fixture；约 55 条（22%）是 sc14 注意到了偏差，却接受或合理化了（'fine'、'defensible'、'edge, ignore'、'宽松一点'），除未交付外每一类都有；明确靠猜测隐藏测试或评分器来决定实现的约 8 条。很多失败不是'没发现'，而是'发现了却放过'。改测试迁就实现的行为也比 evidence_override 的 16 条更普遍：在 opa、obsidian（'Tests I control. Keep normalize'）、kgateway（刷新 golden）、ytt 等处，它作为附带动作出现。
- 与 Kimi 的差距不在于 Kimi 从不犯错。19 条 pattern 写明某个 Kimi rollout 有相同缺陷，集中在边界、状态组合、既有代码假设三类，其中 12 条来自 Baihe fork。另有约 10 条写到 Kimi 在过程中踩了同一个坑，再被自己的测试或探针暴露并修正（katex 空列崩溃、boa kimi-r0、tengo kimi-r8、gocritic kimi-r5、valibot kimi-r20、richlog kimi-r14）。Kimi 的优势主要是四点：更早开始实现并按时 commit；测试从任务原文和外部消费者推导；发现异常后改代码而不是改证据；优先复用仓库约定和已有的权威实现。
- 效率：多份报告记录 sc14 的工具调用更多，但多花在广度和自洽检查上。expr 为 204–353 次，Kimi 为 115–159 次；dateutil 为 171–248 次对 60–74 次，且 sc14 每个 slot 都触发了一次上下文压缩；httpx-multipart 为 126–223 次对 58–91 次；dynamodb 的编译调用为 51–92 次对 15–24 次；scriggo 第一次改 AST 在第 124–286 次调用，Kimi 在第 52–71 次。
- 回归很少：改坏既有行为（P2P 或旧语义）只在约 8 条中出现（clack、kitty、boa、richlog、absslices、bandit-taint、arktype 的 snapshot 等），绝大多数失败发生在新功能的 F2P 测试上。不少 target 的失败只来自一两处具体缺陷，诊断时只改一处就能让整套通过，例如 etree 改一个字段名、kysely 删一个空格、yaegi 加一行 n := n、promreload 改成整数毫秒。
- Kimi 自己也被分析者标出问题：21 条 teacher_contrast（16 个 target）提到 Kimi 用猜测的隐藏测试决定实现范围，18 条（17 个 target）提到 Kimi 断言了没有观察到的事实。这些很少决定评分结果，但说明 Kimi 通过不等于过程干净。

### 按语言（语言按测试文件判定，失败 slot 加权占比；样本小的语言仅供参考）

| 类别 | go (25 个 target) | python (22 个 target) | typescript (17 个 target) | rust (3 个 target) | javascript (1 个 target) |
|---|---|---|---|---|---|
| 自身逻辑在时序/状态/实例/组合上出错 | 17% | 13% | 21% | 20% | 46% |
| 对外接口/格式与契约不符 | 15% | 16% | 12% | 28% | 15% |
| 自证式验证 | 11% | 19% | 15% | 16% | 0% |
| 覆盖/集成缺口 | 9% | 14% | 17% | 8% | 0% |
| 需求误读 | 9% | 15% | 7% | 0% | 31% |
| 未交付 | 15% | 1% | 13% | 16% | 0% |
| 对既有代码/库/框架/语言行为的错误假设 | 11% | 5% | 6% | 12% | 8% |
| 边界值与缺失/空/默认值区分处理错误 | 6% | 11% | 4% | 0% | 0% |
| 无视或篡改反证 | 6% | 6% | 4% | 0% | 0% |

## 方法与可靠性

- 数据：`patterns.jsonl`（全部 315 条，含非主分析）、`primary_patterns.jsonl`（246 条主 pattern）、`analyses.jsonl`（89 份分析）；`extract_patterns.py` 生成。
- 分类（workflow wf_7b43bb70-398）：两个 agent 分别从'根因/行为'和'缺失的工程能力'两个视角独立提出分类，第三个 agent 合并成 9 类并写出判定规则；3 个 agent 分批标注全部 pattern，2 个 agent 再独立标注一遍，最后一个 agent 裁决分歧并写各类说明（`workflow_result.json`）。
- 两轮标注主类 244/246 一致，但合并后的分类说明里按编号点名了不少 pattern，这个一致率偏高。因此另做了一次真正盲测（wf_4e621b04-9ac）：随机抽 60 条，只给去掉所有 pattern 编号和例子的定义与规则（`taxonomy_blind.json`），主类与最终标签一致 54/60（90%，Cohen κ = 0.88），考虑第二标签后 60/60 重合。分歧集中在'状态/组合'与'覆盖/集成缺口'之间。
- 语言：DeepSWE 的 task.toml 有 4 处误标（httpx-deterministic-cookie-store 实为 Python，prometheus-transactional-reload-status 实为 Go，katex-multicolumn-array-spans 与 koota-entity-snapshot-rollback 的测试是 TypeScript），本报告按测试文件判定语言。

## 注意事项

- 数据范围：共 68 个 target，每个只取一份主分析（优先级 official4 > official3 > official2 > official1 > val1 > Baihe fork），另有 21 份早期或重复的分析（official1 旧版、val1、Baihe 的副本）没有计入。被分析的都是 Kimi 能做而 sc14 失败的 target，所以结论描述的是 sc14 相对 Kimi 的差距，不是 sc14 全部失败的分布。sc14 在这些 target 上的通过率：official1/2 和 Baihe fork 多为 0–1/5，official3 为 2–3/5，official4 为 1–5/10。
- 各批次 rollout 数不同。official1–3 和 Baihe fork 每边 5 个 rollout，全部给分析者看（包括 sc14 成功和 Kimi 失败的 slot）。official4（13 个 target、54 条）每边 10 个，但只给分析者看失败的 sc14 和成功的 Kimi（teacher_pass_student_fail）。因此在 official4 里看不到 Kimi 的失败，'Kimi 也有同样缺陷'在那里结构性为 0；频率的分母是失败 slot 数（1–9 个），而不是 5；也没有成功的 sc14 slot 可以对照。
- Baihe fork：13 个 target / 57 条来自 Baihe 早期的 pilot/q 系列，报告格式和 hengyuf 的 official 批次不完全相同。那里 Kimi 只有 3–4/5 通过，所以 19 条'Kimi 共享缺陷'中有 12 条来自这里；而 official1 中 Kimi 全是 5/5。
- 计数不等于独立失败数。同一份分析里的 pattern 互相重叠：自证式验证和部分无视反证 pattern 被分析者明确标为与具体缺陷重叠。n_student_slots 合计 599，远大于实际失败的 slot 数；个别 pattern 引用的 slot 数比失败 slot 还多（如 #3 为 5/4、#72 为 5/4、#191 为 3/2），因为也引用了通过或被排除的 slot。比例应读作'分析者识别出的机制分布'。
- 标注口径：主类按 pattern 的名称和首句判定，忽略分析者自填的 category。两名标注者在主类上 244/246 一致，但 a 就是 taxonomy 作者自己的映射，b 用的也是同一套规则，所以这不是对 taxonomy 可靠性的独立估计。两处分歧已裁决：mobly #143 归为需求误读，tomlkit #231 归为既有代码假设。第二标签取 a 的，a 为空或为 other 时取 b 的，仅供参考。状态组合与覆盖缺口、边界值与需求误读、需求误读与状态组合、接口与覆盖缺口（归属问题）之间仍然模糊，规则稍作调整，各类计数会变动几条。
- 部分接口和需求类失败与规格不够明确有关。分析者在 promreload（int64 精度）、boa（构造函数签名）、sqlite（参数顺序）、igel（detail 的形状）、mobly（组 hook 的错误记录）、ytt（对非数组用 filter）等处注明，任务原文没有写死，隐藏测试或参考实现更严格。这些更多反映 sc14 缺少对惯例的先验，不完全是读错题。反过来，隐藏测试有时比参考实现宽松（tomlkit 中 Kimi 丢弃注释也能通过），所以 Kimi 通过不等于完全符合规格。
- 关键词统计（自测写死缺陷、注意到却接受、猜测隐藏测试、Kimi 共享缺陷、Kimi 中途修正）都是对 pattern 文本的粗扫，是近似值。它们与 taxonomy 说明中的数字（约 63/35/26/17）口径不同，不宜直接比较。
- 这些分析由 codex 读 transcript 并在沙箱里重放得出。很多 pattern 有诊断补丁佐证（'只改一处就通过'），但根因叙述仍是分析者的解读。少数 pattern 并不决定评分（httpx-multipart 编造上游 PR 号、aiomonitor 全命名快照被淘汰但没有测试覆盖），仍然计入了统计。
- 上一步生成 taxonomy 时，有 agent 违反只读要求，在共享的 /tmp 留下了其他用户可读的草稿文件（由 pattern 文本派生，不含密钥）；已于 2026-11-03 删除。

## 文件

`REPORT.md`（本文）、`category_stats.json`、`workflow_result.json`（分类、两轮标注的裁决结果、各类说明）、`taxonomy_blind.json`、`blind_check_labels.json`、`blind_check_agreement.json`、`primary_patterns.jsonl` / `patterns.jsonl` / `analyses.jsonl`、`extract_patterns.py`、`make_report.py`。


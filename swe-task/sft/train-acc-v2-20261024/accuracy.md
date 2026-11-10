# SFT v2 training accuracy vs base (2026-10-24; 10/20-epoch checkpoints added 2026-10-25; empty-think-prefix 2026-10-27; SFT-v3 empty-think-prefix 5/10 ep 2026-10-30)

50 tasks sampled (seed 20261024) from the 97 hengyuf tasks of the v2 SFT pool (Baihe's 97: snapshots not accessible); 5 rollouts per task per model, the synth pipeline's probe harness and each task's grader; every model served with the lenient-think parser.

| model | groups done | rollouts | graded | infra | pass@1 | task-mean | pass@5 | tasks 5/5 |
|---|---|---|---|---|---|---|---|---|
| v2 SFT 5 ep (iter38) | 8/8 | 250 | 249 | 1 | 18.1% | 18.0% | 52.0% | 0 |
| v2 SFT 10 ep (iter76) | 8/8 | 250 | 250 | 0 | 36.0% | 36.0% | 74.0% | 2 |
| v2 SFT 20 ep (iter152) | 8/8 | 250 | 250 | 0 | 44.0% | 44.0% | 74.0% | 4 |
| v2 empty-think-prefix 5 ep (iter38) | 8/8 | 250 | 249 | 1 | 26.9% | 26.8% | 68.0% | 0 |
| v2 empty-think-prefix 10 ep (iter76) | 8/8 | 250 | 249 | 1 | 39.4% | 39.2% | 80.0% | 2 |
| v2 empty-think-prefix 20 ep (iter152) | 8/8 | 250 | 250 | 0 | 52.8% | 52.8% | 84.0% | 7 |
| v3 empty-think-prefix 5 ep (iter71) | 8/8 | 250 | 250 | 0 | 44.0% | 44.0% | 82.0% | 5 |
| v3 empty-think-prefix 10 ep (iter142) | 8/8 | 250 | 245 | 5 | 56.7% | 56.5% | 90.0% | 9 |
| v3 empty-think-prefix 20 ep (iter284) | 8/8 | 250 | 250 | 0 | 54.4% | 54.4% | 92.0% | 5 |
| base (sc14) | 8/8 | 250 | 250 | 0 | 17.2% | 17.2% | 46.0% | 0 |

| task | run | v2 SFT 5 ep (iter38) | v2 SFT 10 ep (iter76) | v2 SFT 20 ep (iter152) | v2 empty-think-prefix 5 ep (iter38) | v2 empty-think-prefix 10 ep (iter76) | v2 empty-think-prefix 20 ep (iter152) | v3 empty-think-prefix 5 ep (iter71) | v3 empty-think-prefix 10 ep (iter142) | v3 empty-think-prefix 20 ep (iter284) | base (sc14) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| syn-aiohttp-cand-c01-r1b4-9f77d81d | official1-httpx | ..P.P | P.PPP | PPPPP | P.PPP | PP.PP | PPPPP | .PP.P | PPPPP | PPPPP | P.... |
| syn-aiohttp-cand-c03-r1b1-b706aac1 | official1-httpx | .PPPP | .PP.P | PPPPP | P.PPP | P..PP | PPPPP | PPPPP | PPPPP | PPP.P | .PPPP |
| syn-aiohttp-cand-c04-7f733601 | official1-httpx | ..... | ..... | ..... | ..... | ..... | ..... | ..... | ..... | ..... | ..... |
| syn-aiohttp-cand-c05-r1b0-7d9ec1f5 | official1-httpx | PPPP. | PPP.P | .PPPP | ...PP | PPP.P | PPPPP | PPPPP | PPPPP | PPPPP | PP.PP |
| syn-argo-cd-cand-c05-cc3e1272 | official1-arcane | ..P.. | P.... | ...P. | ..... | ..... | .P..P | P...P | ...P. | .P.P. | ..... |
| syn-bitecs-cand-c05-r1b0-85a91c3b | official1-koota | P.... | ..PP. | ...PP | PP.PP | P.... | P...P | PP..P | P.P.. | .P.PP | ..... |
| syn-bitecs-cand-c05-r1b4-bc99c654 | official1-koota | P..P. | ....P | PPPP. | ...PP | .PP.. | ..PP. | PPP.. | .PPP. | PPP.. | .P.PP |
| syn-dacite-cand-c01-r1b3-84fe778b | official1-cattrs | .P... | PP... | ..PP. | P..P. | ....P | ..P.. | .P..P | P.P.P | PP.P. | ..... |
| syn-dacite-cand-c02-r1b4-325c5f9d | official1-cattrs | ..... | ....P | PP... | ..... | ..... | ....P | P.... | ...PP | .PPP. | ..... |
| syn-dacite-cand-c04-r1b0-eeb19c3a | official1-cattrs | .PP.P | P.PPP | PP.PP | PP.P. | .PPPP | P.PPP | .PP.. | PPPP. | P..PP | ..... |
| syn-dacite-cand-c05-r1b4-2e37bfb7 | official1-cattrs | .PPPP | P.PPP | PPPPP | .PPP. | PPPPP | PPPPP | PP.PP | PPPPP | PPPP. | ..P.P |
| syn-gomacro-cand-c01-c696c6ca | official1-scriggo | ..... | ..... | ..... | ..... | ...P. | ...P. | ..... | ..... | ..... | ..... |
| syn-gomacro-cand-c02-r1b0-e8804dd9 | official1-scriggo | ..... | ..... | ..... | ..... | ..... | ..... | ..... | ....x | ..P.. | ..... |
| syn-gomacro-cand-c02-r1b3-ed535f70 | official1-scriggo | ..... | ..... | ..... | ..... | ..... | ..... | ..... | ..... | ..... | ..... |
| syn-gomacro-cand-c04-r1b4-2d729ead | official1-scriggo | ..... | ..... | ..... | ...P. | P.... | ..P.. | ..... | ...P. | PP..P | ..... |
| syn-gpython-cand-c04-r1b4-0a5ca664 | official1-anko | ..... | ..... | ..... | ..... | ..... | ..... | ..... | ....P | P..P. | ..... |
| syn-ibis-cand-c03-r1b4-3d3b86b5 | official1-narwhals | ..... | PP... | ...P. | ..P.. | .P.P. | ...P. | ..... | P..x. | ....P | ..... |
| syn-ibis-cand-c04-86a17614 | official1-narwhals | ....P | P.... | ..PPP | ..P.. | P...P | PPPPP | .P.PP | .P.PP | ..PPP | ..... |
| syn-ibis-cand-c04-r1b0-r2b0-35888795 | official1-narwhals | ...PP | .P.PP | PP.PP | .P..P | PP... | .PPPP | ..PPP | PPP.P | .PPPP | P.... |
| syn-ibis-cand-c05-r1b2-r2b3-269e70d9 | official1-narwhals | .P... | .PP.P | .P.PP | P..P. | P..PP | P.PP. | ..PP. | PP.P. | ...PP | .PP.. |
| syn-ibis-cand-c05-r1b3-9f713fee | official1-narwhals | ..... | P.P.. | PPP.P | .P... | .P... | P.PPP | ..PPP | ...P. | P.PPP | ...P. |
| syn-jest-cand-c01-r1b3-08931258 | official1-vitest | ..... | ..... | ..... | P...P | .P... | ..... | P.... | ...P. | ...P. | ..... |
| syn-jest-cand-c04-r1b4-d8b1da89 | official1-testem | ..... | .PPP. | PP.PP | .P..P | PPP.P | P.PP. | .P..P | PPPPP | .PPPP | ..... |
| syn-jest-cand-c05-9e1caaee | official1-testem | .P... | ..... | P.... | .P..P | ...x. | PP... | PPP.. | PPP.P | ....P | ..... |
| syn-jest-cand-c05-r1b2-d6a136fe | official1-vitest | ..... | ..... | ...P. | ..... | .PP.. | .PPPP | ....P | P.PP. | .PP.. | ..... |
| syn-pylama-cand-c01-r1b4-r2b4-68d85058 | official1-bandit | ...PP | ..P.P | .P... | ..P.. | P..P. | ..PPP | P.PP. | .PPPP | PPPPP | PPPP. |
| syn-pylama-cand-c02-r1b0-r2b2-02688723 | official1-bandit | ..... | ..P.. | ..... | ..... | ..... | ..... | P.... | ..... | ..P.. | ..... |
| syn-pylama-cand-c03-r1b2-6182c7b3 | official1-bandit | ...P. | PPPP. | PP.P. | PP..P | PPP.. | PP... | P.PPP | PPPPP | P.PPP | ..... |
| syn-pylama-cand-c03-r1b4-r2b4-58e91fc7 | official1-bandit | ..... | ...P. | ..... | ..... | ..... | ..... | .P..P | ....P | ....P | P.... |
| syn-pylama-cand-c05-r1b4-6eedf9a6 | official1-bandit | .P... | PPPPP | .PPPP | .P... | P.PP. | .PPPP | ...PP | PPP.P | P..PP | P.... |
| syn-risor-cand-c05-r1b2-3718460d | official1-absslices | .P.P. | P.P.. | .P.PP | .PP.. | ..PPP | PP..P | P.... | PP.PP | ..P.. | ..P.. |
| syn-transitions-cand-c03-c910069f | official1-statemachine | x.... | ....P | ..... | P.... | .P.P. | ..PP. | PPP.P | PPPPP | ...PP | ...P. |
| syn-transitions-cand-c04-r1b2-799317b1 | official1-statemachine | ..P.. | ...P. | ....P | P.P.. | P.PP. | PP... | ..PP. | PPP.P | PPP.P | ...P. |
| syn-transitions-cand-c05-r1b2-6a1e0dca | official1-statemachine | P.... | ..P.P | .PPPP | .PP.P | PPP.. | PP.PP | PPPPP | P.PP. | PPPP. | P.P.. |
| syn-vscode-sqltools-cand-c01-r1b0-24c4a31b | official1-sqlformatter | P.P.P | .PPPP | ..PP. | ..... | PPPP. | P.PPP | PPPPP | .PPPP | .PPPP | ..... |
| syn-vscode-sqltools-cand-c02-r1b0-fb4b293a | official1-sqlformatter | ..... | ..... | ...P. | ..... | ...P. | ..... | ..... | ....P | ..P.. | ..... |
| syn-vscode-sqltools-cand-c02-r1b3-7119812a | official1-sqlformatter | .P... | .P... | PP.P. | P.P.. | .P.PP | PP... | P...P | ..P.P | ..... | ..... |
| syn-vscode-sqltools-cand-c04-r1b3-0c604520 | official1-sqlformatter | ..... | .P.PP | ....P | ..P.. | ...P. | PP..P | P.... | P.PP. | PP..P | ..... |
| syn-vscode-sqltools-cand-c05-r1b3-911416cb | official1-sqlformatter | ..... | ..... | P.PPP | ..... | ...P. | ..PPP | .P..P | ...PP | ..P.. | ..... |
| syn-wasmtime-cand-c01-r1b1-82c9e518 | official1-wasmi | ...P. | .PP.P | P.PPP | ...P. | PPP.. | PPPP. | PPP.P | PPPPP | PPPP. | .PPP. |
| syn-wasmtime-cand-c03-353ec720 | official1-wasmi | ..... | ....P | ..... | ..... | .P.P. | PPP.P | ...P. | PxP.P | ....P | ..P.. |
| syn-wasmtime-cand-c04-r1b1-bbb510ca | official1-wasmi | ..... | ..... | .PPPP | ...P. | ..P.P | .PPPP | .P.PP | xx.PP | PP... | ..... |
| syn-wasmtime-cand-c04-r1b4-786cced5 | official1-wasmi | ..... | ..PP. | PP.P. | .P..P | PPPPP | .P.PP | PPPPP | PPPPP | PPP.P | ..... |
| syn-yj-cand-c01-7e8e179e | official1-dasel | ...P. | PPPP. | P.P.P | ..P.P | PPPP. | PPP.P | P.P.P | P.PPP | .PPPP | .P..P |
| syn-yq-cand-c01-r1b4-c75b758f | official1-ytt | P...P | .PP.P | PPP.P | P.PPP | PPP.P | PPPPP | PPPP. | .PPPP | PPPPP | PP... |
| syn-yq-cand-c02-r1b0-0af03743 | official1-ytt | ..P.. | ..... | ..... | x.... | ..P.. | P...P | P.... | P.... | .PPP. | ..... |
| syn-yq-cand-c02-r1b1-2f81d284 | official1-ytt | ..... | ..P.. | PPP.. | ..P.. | PP... | PPPPP | PPP.. | PP... | .PPPP | .P... |
| syn-yq-cand-c02-r1b4-c30118dd | official1-ytt | ..... | .P.PP | ..... | P.... | ....P | .PPP. | ..... | .PP.P | PPPP. | .P.PP |
| syn-yq-cand-c04-r1b4-cae9dfb2 | official1-ytt | ..... | PPPPP | PPPPP | .P... | P.P.. | PP.P. | PP..P | ..PPP | PPPPP | .P... |
| syn-zsa-cand-c02-r1b1-ae1eb794 | official1-dynamodb | ...P. | ..P.. | ..P.P | ..... | ..... | P..PP | ..P.. | P.PP. | PPP.. | .P... |

P = pass, . = fail, x = infra (excluded).

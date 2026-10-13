# A Flexible and Generalized Constant-Potential Framework in i-PI

[English](README.md)

本项目是在 [i-PI 3.3.0](https://github.com/i-pi/i-pi) 基础上开发的恒电势分子动力学扩展。
它保留了 i-PI 的服务器—客户端架构，在核自由度之外引入一个系统级电子自由度，使电子数能够
在分子动力学过程中随目标功函数动态变化。

程序支持分数电子数、双整数端点插值和 Ne 对电极方法，也支持恒电势路径积分分子动力学：**所有 bead 共享同一个电子数，以各 bead 的平均功函数或平均费米能驱动电子自由度。**

当前版本支持通过 i-PI 连接以下第一性原理客户端运行恒电势模拟：

- VASP 5.4.4 + vaspsol++；
- CP2K 2026.1；
- ABACUS 3.11.0-beta10（PW 和 LCAO）。

三个客户端的恒电势版本补丁分别位于：

- `examples/clients/vasp/vasp-5.4.4-ipi-constant-potential.patch`
- `examples/clients/cp2k/cp2k-2026.1-ipi-constant-potential.patch`
- `examples/clients/abacus/abacus-3.11.0-beta10-pr7939-ipi-constant-potential.patch`

示例文件位于 `examples/constant_potential/`。

相关论文：Chenyu Zhang et al., *A Flexible and Generalized Constant-Potential Framework in i-PI*, Journal of Chemical Theory and Computation (2026), [DOI: 10.1021/acs.jctc.6c00504](https://doi.org/10.1021/acs.jctc.6c00504)。

Chenyu Zhang et al., *Electrode Potentials in Computational Electrochemistry: Referencing, Control and Uncertainty*. Current Opinion in Electrochemistry (2026), [DOI: 10.1016/j.coelec.2026.101947](https://doi.org/10.1016/j.coelec.2026.101947)。

## 方法与功能

以总电子数 $q$ 为广义坐标、$p_q$ 为共轭动量。客户端在每次构型求值时接收当前 $q$，完成 SCF 后返回实际电子数、费米能，以及功函数模式所需的功函数。

两种控制模式的电子驱动力为：

$$
F_q=\mu_\star-\overline{\mu},
\qquad
F_q=\overline{\Phi}-\Phi_\star.
$$

其中星号表示目标值，横线表示对 bead 平均。单 bead 时直接使用该构型的反馈。功函数由客户端按 $\Phi=V_{\mathrm{ref}}-E_F$ 计算，$V_{\mathrm{ref}}$ 取指定方向和区间内的平面平均静电势。Ne 模式还可对已接受物理帧的功函数做滑动平均。

恒费米能级控制需要一致的电势基准；参考势保持不变时，固定费米能才对应固定功函数。

电子数计算有两条路径：

- **分数电子数**：一个客户端直接计算当前实数电子数。
- **双整数端点**：两个客户端计算相邻整数电子数，i-PI 使用同一线性权重组合能量、核力、virial、费米能和功函数。电子驱动力采用上述混合反馈定义。

电子热浴可选 Langevin 或 SVR。省略电子热浴的 `temp` 时，它持续跟随体系的物理温度 $T$；多 bead 时仍使用 $T$。显式填写 `temp` 后，电子温度独立设置。

i-PI 的恒电势积分支持 NVE、NVT、NPT、NST；具体计算同时使用客户端支持的晶胞和电子结构设置。电子状态、热浴能量、随机数、功函数窗口和目标扫描进度均随 checkpoint 保存。普通输入不启用 `<electrons>` 时，沿用原有 i-PI 计算流程。

### 电荷补偿策略

恒电势模拟改变总电子数后，电极电子—离子子系统通常带有净电荷。介电极化本身只能重新分布
束缚电荷，在整个周期性晶胞内的积分为零，不能单独保证晶胞电中性。当前示例采用以下补偿策略：

- **VASP + vaspsol++**：`2-vasp-implicit` 使用 `ISOL=2`、`C_MOLAR=1.0` 和
  `R_ION=4.0`，由 Poisson–Boltzmann 连续介质中的移动离子自洽产生补偿电荷。
  `C_MOLAR` 表示体相电解液浓度。
- **ABACUS**：PW 和 LCAO 示例不使用隐式溶剂，使用 `gate_flag=1`、偶极修正和阻挡区，
  由平面 gate 电荷补偿带电电极；gate 的位置及阻挡区必须放在合理的非原子区域。
- **CP2K**：当前 CP2K 所用 implicit Poisson 模型提供介电响应，但没有与 vaspsol++
  `C_MOLAR`/`R_ION` 对应的移动连续介质离子补偿模型。

Ne 对电极是另一种补偿路径。当前实现每次构型求值使用当前电子数及对应的 Ne 有效核电荷；
其数值稳定性依赖电子广义质量、potentiostat 时间常数、SCF 收敛、功函数平均窗口以及 Ne 原子
数目和位置，必须针对具体体系进行收敛性测试。后续可能会根据需要修改为若干步调整一次电子数。

## 安装

### i-PI

使用 Python 3.10 或更高版本，在独立环境中安装：

```bash
git clone https://github.com/zhangchenyu-USTC/i-pi_constant-potential.git
cd i-pi_constant-potential
python -m pip install -e .
i-pi --help
```

也可以在项目目录执行 `source env.sh`，直接使用源码。客户端需编译下列匹配补丁；安装和链接依赖按各软件的构建说明配置。

### VASP 5.4.4 + VASPsol++

在自己的 VASP 5.4.4 源码目录中，依次准备：

1. 官方 `patch.5.4.4.16052018` 修复。
2. VASPsol++ 的 VASP 5.4.4 补丁。
3. 与该版本及编译器配套的 `solvation_intel.F`，复制为 `src/solvation.F`。
4. 本项目的 [VASP 组合补丁](examples/clients/vasp/vasp-5.4.4-ipi-constant-potential.patch)。

```bash
cd /path/to/vasp.5.4.4
patch -p0 < /path/to/patch.5.4.4.16052018
patch -p1 < /path/to/vaspsol++-vasp_5.4.4.patch
cp /path/to/solvation_intel.F src/solvation.F
patch --dry-run -p1 < /path/to/vasp-5.4.4-ipi-constant-potential.patch
patch -p1 < /path/to/vasp-5.4.4-ipi-constant-potential.patch
```

组合补丁包含普通 i-PI 接口，直接按 VASP 构建流程编译即可。普通 socket 使用 `IBRION=23`，恒电势使用 `IBRION=24`。

VASPsol 的更新周期由 INCAR 中新增的 `SOL_UPDATE` 参数控制，取正整数，默认 `1`。例如 `SOL_UPDATE=5` 在本地构型求值编号 `0、5、10…` 更新溶剂响应，其余构型复用冻结的溶剂场，继续进行当前电子数下的电子 SCF，并计算能量、核力和功函数。重启或重新 INIT 后首次刷新并重新计数；STATUS 和重复读取结果不计数。`SOL_UPDATE>1` 的当前使用方式是单 bead、每个端点独立 VASP 进程。其他 VASP 模式使用默认值 `1`。

### CP2K 2026.1

在干净的 CP2K `v2026.1` 源码上应用[完整补丁](examples/clients/cp2k/cp2k-2026.1-ipi-constant-potential.patch)，然后按 CP2K 的 toolchain 或 CMake 流程构建安装：

```bash
cd /path/to/cp2k-2026.1
git apply --check /path/to/cp2k-2026.1-ipi-constant-potential.patch
git apply /path/to/cp2k-2026.1-ipi-constant-potential.patch
```

运行时使用本次安装的程序及其配套共享库，并设置 `CP2K_DATA_DIR` 指向 CP2K 数据目录。Ne 模式需要 `CORE_CORR_DIP` 和 `SURFACE_DIPOLE_CORRECTION`，完整设置见[示例 5](examples/constant_potential/5-cp2k-Ne/cp2k.inp)。

### ABACUS 3.11.0-beta10

补丁以原版 `3.11.0-beta10` 为基线，对应上游提交 `40e7ce8c575abb2ae7fc0df1bd6ec4b48a695d2d`：

```bash
git clone https://github.com/deepmodeling/abacus-develop.git
cd abacus-develop
git checkout 40e7ce8c575abb2ae7fc0df1bd6ec4b48a695d2d
git apply --check /path/to/abacus-3.11.0-beta10-pr7939-ipi-constant-potential.patch
git apply /path/to/abacus-3.11.0-beta10-pr7939-ipi-constant-potential.patch
```

[补丁](examples/clients/abacus/abacus-3.11.0-beta10-pr7939-ipi-constant-potential.patch)扩展原生 socket driver，提供电子数控制、实际占据检查、费米能和功函数反馈。参考势与原生静电势 cube 输出共用计算过程，电荷补偿使用原生 gate。编译方法见 ABACUS 文档；恒电势案例使用固定晶胞。

## 输入与运行

### 先运行一个解析势例子

[示例 1](examples/constant_potential/1-pes-fermi/)不需要 DFT 客户端，可直接检查安装和电子动力学：

```bash
cp -r examples/constant_potential/1-pes-fermi /path/to/pes-run
cd /path/to/pes-run
i-pi input.xml > ipi.log 2>&1
```

下面是一份对应的完整三步输入，在同目录放置示例 1 的 `init.xyz` 即可运行：

```xml
<simulation verbosity="low">
  <output prefix="simulation">
    <properties stride="1" filename="out">[step, time{femtosecond}, potential{electronvolt}, electronic_charge, fermi_level{electronvolt}]</properties>
    <checkpoint filename="restart" stride="1" overwrite="True"/>
  </output>
  <total_steps>3</total_steps>
  <prng><seed>31245</seed></prng>
  <ffdirect name="toy_pes">
    <pes>asym_dw_constant_potential</pes>
    <parameters>{a:1.0, b:3.0, d:1.0, x0:0.0, c:-10.0, pzc:5.0, cap:10.0, q_default:1.0}</parameters>
  </ffdirect>
  <system>
    <initialize nbeads="1">
      <cell mode="abc" units="angstrom">[10.0, 10.0, 10.0]</cell>
      <file mode="xyz" units="angstrom">init.xyz</file>
      <velocities mode="thermal" units="kelvin">300.0</velocities>
    </initialize>
    <forces><force forcefield="toy_pes"/></forces>
    <ensemble><temperature units="kelvin">300.0</temperature></ensemble>
    <motion mode="dynamics">
      <fixatoms_dof>[1, 2]</fixatoms_dof>
      <dynamics mode="nvt" splitting="baoab">
        <timestep units="femtosecond">0.5</timestep>
        <thermostat mode="svr"><tau units="femtosecond">200.0</tau></thermostat>
        <electrons enabled="true">
          <target_fermi_level units="electronvolt">-4.5</target_fermi_level>
          <neutral_electrons>1</neutral_electrons>
          <q_init>1.0</q_init>
          <mass units="atomic_unit">200.0</mass>
          <potentiostat mode="svr"><tau units="femtosecond">200.0</tau></potentiostat>
        </electrons>
      </dynamics>
    </motion>
  </system>
</simulation>
```

### 恒功函数与电子参数

恒电势设置位于 `<motion><dynamics><electrons>`。使用外部 DFT 客户端时，按对应案例设置 `ffsocket` 和客户端连接地址。功函数模式使用以下目标和参考窗口替代费米能目标：

```xml
<target_workfunction units="electronvolt">4.6</target_workfunction>
<potential_average_axis>z</potential_average_axis>
<potential_average_region units="angstrom">[12.5, 15.0]</potential_average_region>
```

参考窗口选在目标真空或溶剂参考区域的静电势平台；更换结构时按平面平均势曲线设置方向和区间。

| 参数 | 含义 |
|---|---|
| `target_fermi_level` / `target_workfunction` | 选择一种控制目标 |
| `neutral_electrons` | 中性体系的总电子数 |
| `q_init`、`p_init` | 初始总电子数及其广义动量 |
| `mass` | 电子数自由度的虚拟质量 |
| `potential_average_axis`、`potential_average_region` | 功函数参考势的方向及平均区间 |
| `charge_mixing="true"` | 在 `<electrons>` 上启用双整数端点插值 |
| `Ne_doping="true"` | 在 `<electrons>` 上启用 Ne 反电极模式 |
| `workfunction_average_steps` | Ne 模式的功函数滑动平均窗口 |
| `potentiostat` | 电子热浴类型、时间常数及可选温度 |

双端点模式定义两个 socket，i-PI 会调用两端并组合结果。端点配置见[示例 4](examples/constant_potential/4-cp2k-implicit-mixing/input.xml)。


### 输出与恢复

`electronic_charge` 输出总电子数 $q$，`fermi_level`、`workfunction` 输出当前反馈；Ne 滑动平均可输出 `workfunction_steps_average`。目标、电子动量和巨正则势也可加入 properties 列表。

恢复时读取生成的 checkpoint。`total_steps` 表示最终步数：将其设为大于 checkpoint 中当前 `step` 的值，然后运行：

```bash
i-pi simulation.restart > ipi-restart.log 2>&1
```

## 案例与验证

| 目录 | 内容 |
|---|---|
| [1-pes-fermi](examples/constant_potential/1-pes-fermi/) | 解析势恒费米能动力学 |
| [2-vasp-implicit](examples/constant_potential/2-vasp-implicit/) | VASPsol++ 隐式电解液恒功函数动力学与 `SOL_UPDATE` |
| [3-vasp-Ne](examples/constant_potential/3-vasp-Ne/) | VASP Ne 反电极恒功函数动力学 |
| [4-cp2k-implicit-mixing](examples/constant_potential/4-cp2k-implicit-mixing/) | CP2K 隐式介电环境，FARMING 双整数端点 |
| [5-cp2k-Ne](examples/constant_potential/5-cp2k-Ne/) | CP2K Ne 分数电子数恒功函数动力学 |
| [6-abacus-pw-compensating-charge](examples/constant_potential/6-abacus-pw-compensating-charge/) | ABACUS PW 平面补偿电荷 |
| [7-abacus-lcao-compensating-charge](examples/constant_potential/7-abacus-lcao-compensating-charge/) | ABACUS LCAO 平面补偿电荷 |


## 引用

使用本项目时，请引用恒电势方法论文及 i-PI 3.0 论文：

```bibtex
@article{zhang2026flexible,
  title   = {A Flexible and Generalized Constant-Potential Framework in i-PI},
  author  = {Zhang, Chenyu and Zhao, Ruoting and He, Zhengda and Iannuzzi, Marcella and Chen, Yanxia and Lan, Jinggang},
  journal = {Journal of Chemical Theory and Computation},
  year    = {2026},
  doi     = {10.1021/acs.jctc.6c00504}
}

@article{litman2024ipi,
  title   = {i-PI 3.0: a flexible and efficient framework for advanced atomistic simulations},
  author  = {Litman, Yair and Kapil, Venkat and Feldman, Yotam M. Y. and Tisi, Davide and Begušić, Tomislav and others},
  journal = {The Journal of Chemical Physics},
  volume  = {161},
  pages   = {062505},
  year    = {2024},
  doi     = {10.1063/5.0215869}
}
```

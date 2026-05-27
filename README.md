# i-PI_constant_potential — A Flexible and Generalized Constant-Potential Molecular Dynamics Framework

This project is an experimental version developed based on i-PI 3.1.5.1. It implements a flexible and generalized constant-potential molecular dynamics framework. The goal of this framework is to decouple constant-potential molecular dynamics simulations from any specific electronic-structure code, enabling them to be combined with different first-principles programs as well as machine-learning potential backends through the server–client communication architecture of i-PI.

The code is currently under development and testing. An official release is coming soon, expected within the next few weeks.

Related paper:

**A Flexible and Generalized Constant-Potential Framework in i-PI**  
https://doi.org/10.1021/acs.jctc.6c00504

---

## Main Idea

This framework is built upon the distinctive server–client architecture of i-PI. Its central idea is to introduce an additional electronic degree of freedom in addition to the conventional nuclear degrees of freedom, and to couple this electronic degree of freedom to a target potential, such as the Fermi level or the work function. The propagation of the electronic degree of freedom, the constant-potential control, and the exchange of electronic information such as the electron number, Fermi level, or work function are all handled on the i-PI side.

As a result, the method is not tied to any specific electronic-structure package. Instead, it can be interfaced with different electronic-structure codes through minimal and non-intrusive modifications to the external DFT clients. This design keeps the core constant-potential algorithms mainly within i-PI, thereby improving the portability, flexibility, and extensibility of the method.

The framework is applicable not only to conventional constant-potential simulations based on implicit solvent, but also to a Ne-electrode, or Ne counter-electrode, based constant-potential scheme. In this way, constant-potential simulations can be extended to more general electrochemical interface models, rather than being restricted to implicit-solvent environments.

At the same time, the framework is compatible with two broad classes of electronic-structure backends. For DFT clients that support fractional electron numbers, constant-potential sampling can be performed directly by adjusting the electron number. For DFT clients that are restricted to integer electron numbers, the framework employs a linear interpolation strategy, also referred to as two-endpoint mixing, between two neighboring constant-charge states. By constructing an equivalent mixed-Hamiltonian description, this approach bypasses the integer-electron-number constraint and enables flexible constant-potential sampling within an integer-charge calculation framework.

More broadly, this linear-interpolation strategy establishes a natural bridge between constant-charge and constant-potential simulations. Therefore, the framework is not only applicable to conventional DFT-based constant-potential simulations, but also provides a direct and natural route toward the future development of constant-potential machine-learning potentials.

---

## Key Features

- Introduces an additional electronic degree of freedom on top of i-PI to enable constant-potential molecular dynamics simulations.

- Uses the server–client communication architecture of i-PI, with the constant-potential control algorithms mainly handled on the i-PI side, providing good generality and portability.

- Can be coupled to multiple DFT clients. Interface patches for VASP and CP2K are currently provided, and communication interfaces for additional DFT clients are under development.

- Supports constant-potential control using physical quantities such as the Fermi level or the work function as the target potential.

- Provides two stochastic potentiostats: CSVR and Langevin. Compared with deterministic schemes such as Nosé–Hoover methods, stochastic thermostats typically provide better dynamical properties and ergodicity for a single electronic degree of freedom. In test simulations, the electron number exhibits a stable and physically reasonable equilibrium distribution.

- Supports both conventional implicit-solvent constant-potential simulations and Ne-electrode, or Ne counter-electrode, based constant-potential simulation schemes.

- Enables direct constant-potential sampling for electronic-structure programs that support fractional electron numbers.

- Enables constant-potential sampling for electronic-structure programs restricted to integer electron numbers through linear interpolation between two neighboring constant-charge endpoints.

- Establishes a connection between constant-charge and constant-potential simulations through the linear-interpolation strategy, making the method compatible with electronic-structure backends at different levels.

- Can be naturally extended to constant-potential machine-learning potentials, providing a foundation for future large-scale constant-potential molecular dynamics simulations.

---

## Usage

To use this version of i-PI, the external electronic-structure clients must first be patched accordingly. The current implementation relies on the client patches provided in the following directories:

```text
examples/clients/vasp
examples/clients/cp2k
```

Therefore, before running constant-potential molecular dynamics simulations, the corresponding patches need to be applied to the VASP or CP2K client, followed by recompilation and reinstallation of the electronic-structure code.

After the patched client has been installed, the framework can be used in essentially the same way as standard i-PI. Users can prepare the i-PI input files as usual, launch the i-PI server, and connect the patched VASP or CP2K client to perform constant-potential molecular dynamics simulations.

Example input files are available in:

```text
examples/constant-potential/
```

These examples provide typical setups for constant-potential simulations and can serve as starting points for practical calculations. Users may modify the input files according to the specific system, target potential, electronic-structure client, and desired constant-potential control scheme.

---

## Notes

This project is currently an experimental development version and is mainly intended for method testing, functional validation, and reproduction of the results reported in the related paper. Different electronic-structure clients may treat the electron number, Fermi level, work function, and charged systems in different ways. Therefore, before practical use, users are advised to carefully check the client patches, input-file settings, and the definition of the target potential for consistency with the specific research system.

For DFT clients restricted to integer electron numbers, the linear-interpolation strategy relies on information from two neighboring constant-charge states. It is therefore important to ensure that the calculation settings for the two endpoints are consistent, including the structure, computational parameters, energy convergence criteria, and electronic-structure settings.

For Ne-electrode-based constant-potential simulations, the position of the Ne counter electrode, the charge-compensation scheme, and the corresponding potential definition should be chosen carefully according to the system geometry and research objective.

---

## Citation

If you use this model or code in your research, please cite the following paper:

```bibtex
@article{zhang2026flexible,
  title={A Flexible and Generalized Constant-Potential Framework in i-PI},
  author={Zhang, Chenyu and Zhao, Ruoting and He, Zhengda and Iannuzzi, Marcella and Chen, Yanxia and Lan, Jinggang},
  journal={Journal of Chemical Theory and Computation},
  year={2026},
  doi={10.1021/acs.jctc.6c00504}
}
```

Please also cite the relevant i-PI paper:

```bibtex
@article{litman2024pi,
  title={i-PI 3.0: A flexible and efficient framework for advanced atomistic simulations},
  author={Litman, Yair and Kapil, Venkat and Feldman, Yotam MY and Tisi, Davide and Begu{\v{s}}i{\'c}, Tomislav and Fidanyan, Karen and Fraux, Guillaume and Higer, Jacob and Kellner, Matthias and Li, Tao E and others},
  journal={The Journal of Chemical Physics},
  volume={161},
  number={6},
  year={2024},
  publisher={AIP Publishing}
}
```

---

## License and Release Notes

This project is currently released as experimental research code to support the development of constant-potential molecular dynamics methods and the reproduction of the results reported in the related paper. After the official release, the installation instructions, example files, client patches, interface documentation, and license information will be further improved.

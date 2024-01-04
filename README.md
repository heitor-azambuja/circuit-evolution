# circuit-evolution
Evolve circuits components values with GA.

This code was developed and tested on Ubuntu 20 LTS and Debian 12. It should work on Windows, but installation of Ngspice might be different, check [PySpice documentation](https://pyspice.fabrice-salvaire.fr/releases/v1.4/installation.html#on-windows).

## Installation
Install Ngspice
```bash
sudo apt-get -y install ngspice libngspice0 libngspice0-dev
```

Create virtual environment
```bash
python3 -m venv venv
```

Activate virtual envitonment
```bash
source venv/bin/activate
```

Install python requitements
```bash
python3 -m pip install -r requirements.txt
```

Test PySpice installation
```bash
pyspice-post-installation --check-install
```

## Modules
A brief description of all the modules in the project.

### components.spice
This file contains the Spice models of NPN and PNP transistors that are used in the circuits.

### components.py
This file contains a class that defines a digital potenciomenter.

### circuits.py
Classes to instantiate the circuits that are evolved. Two BJT class A amplifiers, one with 4 resistors, ant one with 8 resistors.

### amp_4r_evolution.py And amp_8r_evolution.py
Scripts that define the fitness function and run the evolution of the amplifier with 4 and 8 resistors respectively.

## Running
Evolve class A amplifier with 4 resistors:
```bash
python3 amp_4r_evolution.py
```
Evolve class A amplifier with 8 resistors:
```bash
python3 amp_8r_evolution.py
```

When running, Pyspice might show some warning and error messages about Ngspice not supported version. But it should work fine.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

<!-- Please make sure to update tests as appropriate. -->

## Resources

- [PySpice](https://pyspice.fabrice-salvaire.fr/releases/v1.4/overview.html)
- [NgSpice](http://ngspice.sourceforge.net/)
- [PyGad](https://pygad.readthedocs.io/en/latest/)
- PySpice introduction [video playlist](https://youtube.com/playlist?list=PL97KTNA1aBe1QXCcVIbZZ76B2f0Sx2Snh&si=VEu1G6qsrAJ8VWXl) by [EngineeringThings](https://www.youtube.com/@engineeringthings)

## Author
- Heitor Teixeira de Azambuja - heitortazamba@gmail.com - [![Linkedin](https://i.stack.imgur.com/gVE0j.png)](https://www.linkedin.com/in/heitor-azambuja/)

## License

[MIT](LICENSE)
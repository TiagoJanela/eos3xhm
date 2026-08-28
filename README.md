# HADES Oral Drug-Likeness

Scores how closely a compound resembles an approved oral drug, with 0.63 the authors' recommended cutoff. HADES averages probabilities from five tree and boosting classifiers over 298 features combining Mordred descriptors, ADMET-AI predictions and QED terms, trained on 1,177 approved oral drugs against 5,307 nondrugs from ChEMBL, ZINC and GDB. Scores rise across clinical phases and fall for orally toxic, small-ring and chemically implausible structures. This model is based on a random-split provided by the authors.



## Information
### Identifiers
- **Ersilia Identifier:** `eos3xhm`
- **Slug:** `hades-oral-druglikeness`

### Domain
- **Task:** `Annotation`
- **Subtask:** `Property calculation or prediction`
- **Biomedical Area:** `Any`
- **Target Organism:** `Any`
- **Tags:** `Drug-likeness`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `1`
- **Output Consistency:** `Fixed`
- **Interpretation:** Probability that a compound resembles an approved oral drug; values above 0.63 indicate likely oral drug-likeness.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| hades_score | float | high | Probability that the molecule resembles an approved oral drug with 0.63 the recommended cutoff |


### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`

### Resource Consumption


### References
- **Source Code**: [https://github.com/Narek-Petros-yan/HADES](https://github.com/Narek-Petros-yan/HADES)
- **Publication**: [https://doi.org/10.1021/acs.jcim.5c02953](https://doi.org/10.1021/acs.jcim.5c02953)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2026`
- **Ersilia Contributor:** [TiagoJanela](https://github.com/TiagoJanela)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [MIT](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos3xhm
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos3xhm
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!

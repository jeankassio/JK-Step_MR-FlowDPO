# Attribution and scope

JK-Step MR-FlowDPO is a modified distribution of [Side-Step](https://github.com/koda-dernet/Side-Step), by koda-dernet, initially based on commit `fc800935a4b7f13a05103eafb7bacd623d7ed4f7`.
Changes include ACE-Step preference training, multi-reward dataset pairing, a new CLI/browser interface, automatic model downloads, installation and documentation.

The inherited **CC BY-NC-SA 4.0** license in `LICENSE` applies to this distribution. Redistribution must preserve attribution and the license; commercial use is not granted by that license. Dependencies, model weights and datasets keep their respective licenses. The old upstream package metadata said MIT; JK-Step's package metadata has been corrected to the license actually shipped with the source.

The ACE-Step runtime code and model architecture are from [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) and its official model repositories. Pure SFT XL weights are downloaded from [jeankassio/acestep_v1.5_sft_xl](https://huggingface.co/jeankassio/acestep_v1.5_sft_xl). The implementation of the preference loss follows the equations in [MR-FlowDPO](https://arxiv.org/abs/2512.10264), rather than copying the authors' Audiocraft trainer. No affiliation or endorsement by those projects is implied.

"""Localize user-facing API errors without modifying data or diagnostics."""
from __future__ import annotations

_MESSAGES = [
    ('Execução não encontrada', 'Run not found', 'Ejecución no encontrada'),
    ('Operação desconhecida', 'Unknown operation', 'Operación desconocida'),
    ('Já existe uma execução MR-FlowDPO ativa', 'An MR-FlowDPO job is already active', 'Ya hay una ejecución MR-FlowDPO activa'),
    ('Outra operação está ativa na interface SFT/datasets', 'Another operation is active in the SFT/dataset interface', 'Otra operación está activa en la interfaz SFT/datasets'),
    ('Pasta de músicas não encontrada', 'Music folder not found', 'Carpeta de música no encontrada'),
    ('Opções precisam ser um objeto JSON', 'Options must be a JSON object', 'Las opciones deben ser un objeto JSON'),
    ('O servidor reiniciou enquanto o processo continuou ativo. Encerre esse processo antes de iniciar outra execução.', 'The server restarted while the process remained active. Stop that process before starting another run.', 'El servidor se reinició mientras el proceso seguía activo. Detén ese proceso antes de iniciar otra ejecución.'),
    ('Use um nome de até 100 caracteres', 'Use a name of up to 100 characters', 'Usa un nombre de hasta 100 caracteres'),
    ('A configuração precisa ser um objeto JSON', 'Configuration must be a JSON object', 'La configuración debe ser un objeto JSON'),
    ('Configuração deve ser um objeto JSON', 'Configuration must be a JSON object', 'La configuración debe ser un objeto JSON'),
    ('Use um arquivo JSON de candidatos/avaliações', 'Use a candidate/ratings JSON file', 'Usa un archivo JSON de candidatos/evaluaciones'),
    ('O editor aceita JSON até 20 MB. Divida os candidatos ou edite o arquivo externamente.', 'The editor accepts JSON up to 20 MB. Split the candidates or edit the file externally.', 'El editor acepta JSON de hasta 20 MB. Divide los candidatos o edita el archivo externamente.'),
    ('Arquivo precisa conter um objeto com lista samples', 'The file must contain an object with a samples list', 'El archivo debe contener un objeto con una lista samples'),
    ('Esse diretório não contém training_state.pt; use init_adapter para um LoRA sem estado de retomada.', 'This directory has no training_state.pt; use init_adapter for a LoRA without resume state.', 'Este directorio no contiene training_state.pt; usa init_adapter para un LoRA sin estado de reanudación.'),
    ('training_config.json precisa conter um objeto JSON', 'training_config.json must contain a JSON object', 'training_config.json debe contener un objeto JSON'),
    ('Use um arquivo .json com lista samples', 'Use a .json file with a samples list', 'Usa un archivo .json con una lista samples'),
    ('Manifesto não encontrado', 'Manifest not found', 'Manifiesto no encontrado'),
    ('Manifesto precisa conter uma lista de pares', 'The manifest must contain a list of pairs', 'El manifiesto debe contener una lista de pares'),
    ('Dataset não encontrado', 'Dataset not found', 'Dataset no encontrado'),
    ('Dataset precisa conter uma lista de amostras', 'The dataset must contain a list of samples', 'El dataset debe contener una lista de muestras'),
    ('Use um relatório de preparação JSON de até 20 MB', 'Use a preparation report JSON of up to 20 MB', 'Usa un informe de preparación JSON de hasta 20 MB'),
    ('Esse arquivo não é um relatório de preparação', 'This file is not a preparation report', 'Este archivo no es un informe de preparación'),
    ('Lista de exclusões inválida', 'Invalid exclusions list', 'Lista de exclusiones no válida'),
    ('Amostra inválida', 'Invalid sample', 'Muestra no válida'),
    ('Amostra não encontrada', 'Sample not found', 'Muestra no encontrada'),
    ('Áudio não encontrado', 'Audio not found', 'Audio no encontrado'),
    ('Par ou ramo inválido', 'Invalid pair or branch', 'Par o rama no válido'),
    ('Par não encontrado', 'Pair not found', 'Par no encontrado'),
    ('A preparação terminou sem um dataset de treinamento válido. Consulte as exclusões e o log.', 'Preparation ended without a valid training dataset. Check exclusions and the log.', 'La preparación terminó sin un dataset de entrenamiento válido. Revisa las exclusiones y el registro.'),
]
_EXACT = {pt: {'en': en, 'pt': pt, 'es': es} for pt, en, es in _MESSAGES}
_PREFIXES = [
    ('Campo obrigatório: ', 'Required field: ', 'Campo obligatorio: '),
    ('Não foi possível iniciar: ', 'Could not start: ', 'No se pudo iniciar: '),
    ('Não foi possível interromper o download: ', 'Could not stop the download: ', 'No se pudo detener la descarga: '),
]


def language_from_header(value: str = '') -> str:
    for item in value.split(','):
        code = item.split(';')[0].strip().lower().split('-')[0]
        if code in ('en', 'pt', 'es'):
            return code
    return 'en'


def localize_error(detail, accept_language: str = ''):
    if not isinstance(detail, str):
        return detail
    language = language_from_header(accept_language)
    if detail in _EXACT:
        return _EXACT[detail][language]
    for pt, en, es in _PREFIXES:
        if detail.startswith(pt):
            return {'en': en, 'pt': pt, 'es': es}[language] + detail[len(pt):]
    import re
    match = re.fullmatch(r'O processo terminou com código (-?\d+)\. Consulte o log\.', detail)
    if match:
        return {'en': 'The process ended with code {code}. Check the log.',
                'pt': 'O processo terminou com código {code}. Consulte o log.',
                'es': 'El proceso terminó con código {code}. Revisa el registro.'}[language].format(code=match[1])
    return detail

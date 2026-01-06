"""Script de mantenimiento para limpiar archivos temporales del proyecto.

Elimina carpetas __pycache__ y archivos .log sin tocar models/, docs/ o src/.
Útil antes de commits finales para mantener el repositorio limpio.
"""
from __future__ import annotations

import sys
from pathlib import Path
import shutil


def cleanup_project(root: Path) -> tuple[int, int]:
    """Limpia archivos temporales del proyecto.
    
    Returns:
        tuple[int, int]: (pycache_dirs_removed, log_files_removed)
    """
    pycache_count = 0
    log_count = 0
    
    # Eliminar carpetas __pycache__
    for pycache_dir in root.rglob("__pycache__"):
        if pycache_dir.is_dir():
            print(f"🗑️  Eliminando: {pycache_dir.relative_to(root)}")
            shutil.rmtree(pycache_dir)
            pycache_count += 1
    
    # Eliminar archivos .log
    for log_file in root.rglob("*.log"):
        if log_file.is_file():
            print(f"🗑️  Eliminando: {log_file.relative_to(root)}")
            log_file.unlink()
            log_count += 1
    
    return pycache_count, log_count


def main() -> int:
    """Ejecuta limpieza del proyecto."""
    
    root = Path(__file__).parent
    print(f"🧹 Limpiando proyecto en: {root}\n")
    
    # Verificar que estamos en el directorio correcto
    if not (root / "src").exists():
        print("❌ Error: No se encuentra la carpeta 'src'. ¿Estás en el directorio correcto?")
        return 1
    
    pycache_count, log_count = cleanup_project(root)
    
    print(f"\n✅ Limpieza completada:")
    print(f"   - {pycache_count} carpetas __pycache__ eliminadas")
    print(f"   - {log_count} archivos .log eliminados")
    
    # Verificar que models/, docs/ y src/ siguen intactos
    critical_dirs = ["models", "docs", "src"]
    for dir_name in critical_dirs:
        if (root / dir_name).exists():
            print(f"   ✓ {dir_name}/ preservado")
        else:
            print(f"   ⚠️  {dir_name}/ no encontrado (puede ser normal)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

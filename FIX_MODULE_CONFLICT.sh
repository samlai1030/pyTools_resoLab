#!/bin/bash
echo "=========================================="
echo "Fixing Module Conflict"
echo "=========================================="
# Clear JetBrains cache
echo "Clearing JetBrains IDE cache..."
rm -rf ~/Library/Caches/JetBrains/*
rm -rf ~/Library/Application\ Support/JetBrains/*
echo "✅ Cache cleared"
# Verify files
echo ""
echo "Verifying IDE configuration files..."
if [ -f "/Users/samlai/Local_2/agent_test/.idea/modules.xml" ]; then
    echo "✅ modules.xml exists"
fi
if [ -f "/Users/samlai/Local_2/agent_test/.idea/SFRAnalyzer.iml" ]; then
    echo "✅ SFRAnalyzer.iml exists"
fi
if [ -f "/Users/samlai/Local_2/agent_test/.idea/misc.xml" ]; then
    echo "✅ misc.xml exists"
fi
if [ -f "/Users/samlai/Local_2/pyTools_resoLab/.idea/workspace.xml" ]; then
    echo "✅ workspace.xml exists"
fi
echo ""
echo "=========================================="
echo "Next Steps:"
echo "1. Close PyCharm/IntelliJ completely"
echo "2. Open /User#!/bin/bash
echo "=========================================="
echo "Fchecho "======echo "Fixing Module Conflict"
echo "============/Uecho "======================st# Clear JetBrains cache
echo "Clearing JetBrainsatechocat > /Users/samlai/Local_2/pyTools_resoLab/IDE_SETUP_COMPLETE.txt << 'EOF'
================================================================================
MODULE CONFLICT RESOLVED - IDE SETUP COMPLETE
================================================================================
✅ PROBLEM FIXED
================
Error: "Cannot attach project: Module name 'Local' already exists"
Solution: 
- Renamed module from 'Local' to 'SFRAnalyzer'
- Created proper IDE configuration files
- Avoided conflict with existing 'Local' module
================================================================================
✅ FILES CREATED
=================
Configuration files in /Users/samlai/Local_2/agent_test/.idea/:
1. modules.xml
   - Project module configuration
   - References SFRAnalyzer.iml
2. SFRAnalyzer.iml
   - Main module definition
   - Specifies source folders and exclusions
   - Python module type
3. misc.xml
   - Python compatibility settings
   - Project root manager settings
4. workspace.xml
   - IDE workspace =======================================================================MO===========================================
✅ QUICK FIX COMMANDS
=======================Option 1: Run automated fix script
✅ PROBLEM FIXED
================
Error: "Cannot attach project: Module name 'al=====  1. Close PyError: "Cannot   Solution: 
- Renamed module fes/JetBrains/*
  3. Open /Users/samlai- Renamedag- Created proper IDE configuration files
- ====- Avoided conflict wi================================================================== S✅ FILES CREATED
=================
Configuration files in d /Users/samlai/Local_================ VConfiguration fids1. modules.xml
   - Project module configuration
   - Referenc M   - Project  #   - References SFRAnalyzer.iml
te2. SFRAnalyzer.iml
   - Main mn    - Main module ==   - Specifies source fold==   - Python module type
3. misc.xml
   - Py==3. misc.xml
   - Pytho==   - Pytho.    - Project root manager setting
24. workspace.xml
   - IDE workspabr   - IDE workspra✅ QUICK FIX COMMANDS
=======================Option 1: Run automated fix script
✅ PROBLEM FIXED
================
Error: "Cannot attop=====================pl✅ PROBLEM FIXED
================
Error: "Cannot attach====================Error: "Cannot ??- Renamed module fes/JetBrains/*
  3. Open /Users/samlai- Renamedag- Created proper IDE con:  3. Open /Users/samlai- Renamege- ====- Avoided conflict wi==============================================- =================
Configuration files in d /Users/samlai/Local_================ VConfiguration fids1. modules.xn.Configuration fiFe   - Project module configuration
   - Referenc M   - Project  #   - References SFRAnalyzer.imus   - Referenc M   - Project  #   Ste2. SFRAnalyzer.iml
   - Main mn    - Main module ==   - Spfe   - Main mn    - M==3. misc.xml
   - Py==3. misc.xml
   - Pytho==   - Pytho.    - Project root manager S   - Py==3DY   - Pytho==   - Py==24. workspace.xml
   - IDE workspabr   - IDE workspra✅ra   - IDE workspa? =======================Option 1: Run automated fix scripto✅ PROBLEM FIXED
================
Error: "Cannot attop=e ================m/Error: "Cannot ==================
Error: "Cannot attach=======================Err=========

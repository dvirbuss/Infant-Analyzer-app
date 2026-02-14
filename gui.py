# gui.py
from tkinterdnd2 import TkinterDnD
from ui.gui_ui import InfantAnalyzerGUI

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = InfantAnalyzerGUI(root)
    root.mainloop()

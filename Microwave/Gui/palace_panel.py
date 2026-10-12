# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""The dialog that runs a study on Palace.

Everything here is presentation. ``Gui/palace_steps.py`` holds the steps of a
run, which a run started from the command line takes as well. This module
decides what to show and when, and owns no physics.

A run is the pipeline's stages. The first reads the document and writes
its shapes, so it runs here, on the thread the document belongs to. The mesh and
the solve take minutes and touch only files and processes, so they run on a
worker thread and report by signal. Qt widgets are touched only on the main
thread.

What this panel shows differs from the other backend's for a reason. There is no
grid to draw before a run: the mesh is made by Gmsh from the drawing, and Mesh
here makes it, says what it came out as and puts it in the study without
solving. Run puts the mesh it solved there too. There is one run rather than one
per port: Palace drives every port marked as a source in one solve.
"""

import html

from PySide import QtCore, QtWidgets

from . import fem_mesh
from . import results as results_glue
from .palace_steps import STARTS, PalaceSteps, default_directory
from .plot_s_params import show_matrix
from .run_panel import JobWorker, RunPanel, log_view, version_line
from .run_steps import directory_line


class PalaceTaskPanel(PalaceSteps, RunPanel):
    """Check, Mesh, Run, and each answer kept as a run of its own."""

    def __init__(self, analysis):
        self.analysis = analysis
        self.form = QtWidgets.QWidget()
        self.worker = None
        #: Whether the job on the worker ends in a solve, and so in a matrix,
        #: rather than in a mesh and a report of it.
        self.solving = False
        #: The mesh the job on the worker made, once it has made one. Set on the
        #: worker and read once the job has finished, so a run whose solve
        #: failed still has the mesh it was solved on.
        self.made = None
        #: Colour of the last status set. See ``RunPanel.set_status``.
        self.status_color = None
        #: Every line the log holds, which a run keeps as its own.
        self.logged = []
        self.setup_ui()

    # ------------------------------------------------------------------ UI

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self.form)
        # Escaped, because a Label is whatever the user typed, and Qt reads one
        # holding "<" as markup.
        label = html.escape(self.analysis.Label)
        layout.addWidget(QtWidgets.QLabel(f"<b>Analysis:</b> {label}"))
        self.label_solver = QtWidgets.QLabel()
        layout.addWidget(self.label_solver)
        self.label_rundir = QtWidgets.QLabel()
        layout.addWidget(self.label_rundir)
        self.label_mesh = QtWidgets.QLabel()
        self.label_mesh.setToolTip(
            "Made by Gmsh at each Mesh and Run, and shown in the study as " + fem_mesh.label(STARTS)
        )
        layout.addWidget(self.label_mesh)
        layout.addSpacing(10)

        self.status_label = QtWidgets.QLabel()
        layout.addWidget(self.status_label)
        self.log_view = log_view()
        layout.addWidget(self.log_view)

        row = QtWidgets.QHBoxLayout()
        self.btn_mesh = QtWidgets.QPushButton("Mesh")
        self.btn_mesh.setToolTip(
            "Mesh the study with Gmsh, report the mesh and add it to the study. "
            "Does not start Palace."
        )
        self.btn_check = QtWidgets.QPushButton("Check")
        self.btn_check.setToolTip(
            "Check the model for errors, find Palace and read its version. Does not mesh or solve."
        )
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        #: Every widget this panel takes a signal from, and what it drives.
        self.buttons = (
            (self.btn_mesh, self.on_mesh),
            (self.btn_check, self.on_check),
            (self.btn_run, self.on_run),
            (self.btn_stop, self.on_stop),
        )
        for button, slot in self.buttons:
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)

        version = QtWidgets.QLabel(version_line())
        version.setAlignment(QtCore.Qt.AlignRight)
        layout.addSpacing(4)
        layout.addWidget(version)

        self.set_status("Idle")
        self.update_labels()
        self.update_mesh_label()
        self.set_ui_busy(False)

    def update_labels(self):
        solver = self.solver
        if solver is None:
            self.label_solver.setText("<b>Solver:</b> (none)")
            self.label_rundir.setText("<b>Directory:</b> (none)")
            return
        name = html.escape(solver.Label)
        self.label_solver.setText(
            f"<b>Solver:</b> {name}" + ("" if solver.Label == STARTS else f" ({STARTS})")
        )
        self.label_rundir.setText(
            directory_line(
                str(solver.RunDirectory), default_directory(self.analysis.Document.FileName)
            )
        )

    def update_mesh_label(self):
        """Say whether the mesh in the study still describes it.

        The mark is read first and the study only where the mark says the mesh
        still stands, as the other backend's panel does for its preview. An
        edit to the study can be made anywhere, and the mark is written by the
        edit rather than by this panel.
        """
        reason = fem_mesh.staleness(self.analysis, STARTS)
        if reason is None:
            mesh = fem_mesh.find(self.analysis, STARTS)
            self.label_mesh.setText(
                f"<b>Mesh:</b> {mesh.FemMesh.VolumeCount:,} elements, matches the study"
            )
        else:
            self.label_mesh.setText(f"<b>Mesh:</b> <font color='orange'>{reason}</font>")

    def set_ui_busy(self, busy):
        # Mesh and Run both write into a directory beside the document, so an
        # unsaved document has nowhere for either to go.
        saved = bool(self.analysis.Document.FileName)
        self.btn_mesh.setEnabled(not busy and saved)
        self.btn_check.setEnabled(not busy)
        self.btn_run.setEnabled(not busy and saved)
        self.btn_stop.setEnabled(busy)
        self.btn_run.setToolTip(
            "" if saved else "Save the document first. Runs are written beside it."
        )

    # --------------------------------------------------------------- check

    # ------------------------------------------------------------ mesh, run

    def on_mesh(self):
        self.start(solving=False)

    def on_run(self):
        self.start(solving=True)

    def start(self, solving):
        """Prepare here, then mesh - and solve, if ``solving`` - on the worker."""
        self.release_worker()
        job = self.prepare(solving)
        if job is None:
            return
        self.worker = JobWorker(job, self.explain)
        for signal, slot in self.worker_slots().items():
            getattr(self.worker, signal).connect(slot)
        self.set_ui_busy(True)
        self.set_status("Meshing..." if not solving else "Meshing, then solving...")
        self.worker.start()

    def worker_slots(self):
        return {
            "said": self.log,
            "failed": self.on_failed,
            "finished_signal": self.on_finished,
        }

    def on_finished(self, succeeded):
        self.set_ui_busy(False)
        if self.was_stopped():
            self.set_status("Stopped - nothing from this run is kept", "orange")
            return
        self.finish(succeeded, self.worker.answer)

    def collect(self, answer, made):
        """File the run and its mesh, then show the matrix.

        Filed before it is plotted: a chart is a window that gets closed, and
        the matrix cost minutes.
        """
        filed = super().collect(answer, made)
        if filed is None:
            return None
        matrix, holder = filed
        try:
            if not self.show_refined(holder):
                show_matrix(matrix, results_glue.source(holder), holder.Document.Name)
        except Exception as error:
            self.set_status(f"Plotting failed: {error}", "red")
        return filed

    def on_failed(self, message):
        self.fail("Run failed - see the log", message)

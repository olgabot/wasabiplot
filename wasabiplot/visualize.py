# -*- coding: utf-8 -*-
from __future__ import print_function

from collections import Counter
import warnings

import HTSeq
import matplotlib.pyplot as plt
from matplotlib.artist import Path
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd


INSERTION_DELETIONS = ('I', 'D')
COVERAGE_CIGAR = ('M', )
JUNCTION_CIGAR = ('N', )

TEXT_KWS = dict(#fontsize=6,
                horizontalalignment='center',
                verticalalignment='center',
                backgroundcolor='w')
PATCH_KWS = dict(facecolor='none',)


class WasabiPlotter(object):

    def __init__(self, bam_filename, chrom, start, stop, strand, log_base,
                 color, bad_cigar=INSERTION_DELETIONS,
                 coverage_cigar=COVERAGE_CIGAR, junction_cigar=JUNCTION_CIGAR,
                 warn_skipped=True):
        self.bam_filename = bam_filename
        self.chrom = chrom
        self.start = start
        self.stop = stop
        self.strand = strand
        self.log_base = log_base
        self.color = color
        self.bad_cigar = bad_cigar
        self.coverage_cigar = coverage_cigar
        self.junction_cigar = junction_cigar
        self.warn_skipped = warn_skipped

        self.length = self.stop - self.start + 1
        self.coordinates = self.chrom, self.start, self.stop, self.strand
        self.interval = HTSeq.GenomicInterval(*self.coordinates)

        self.bam = HTSeq.BAM_Reader(self.bam_filename)

        self.coverage = self.count_coverage()
        self.junctions = self.count_junctions()

    def skip_bad_cigar(self, read):
        """Return None if the read has improper CIGAR strings

        Parameters
        ----------
        read : HTSeq.SAM_Alignment
            A single read from a genomic alignment file
        """
        # Skip reads with no CIGAR string
        if read.cigar is None:
            if self.warn_skipped:
                warnings.warn(
                    "Skipping read with no CIGAR string: {read_name} (from "
                    "{bam})".format(read_name=read.read.name,
                                    bam=self.bam_filename))
            return

        # Check if the read contains an insertion (I)
        # or deletion (D) -- if so, skip it
        for cigar_operation in read.cigar:
            cigar = cigar_operation.type
            if cigar in self.bad_cigar:
                if self.warn_skipped:
                    warnings.warn(
                        "Skipping read with CIGAR string {abbrev} (a base in "
                        "the read was {full}): {read_name} "
                        "(from {bam})".format(
                            read_name=read.read.name, bam=self.bam_filename,
                            abbrev=cigar,
                            full=HTSeq.cigar_operation_names[cigar]))
                return
        return read

    def count_coverage(self):
        """Get the number of reads that matched to the reference sequence

        Returns
        -------
        counts : numpy.array
            Number of reads that matched to the genome at every location
        """
        counts = np.ones(shape=self.length, dtype=int)

        region_reads = self.bam[self.interval]

        for read in region_reads:
            read = self.skip_bad_cigar(read)
            if read is None:
                continue
            for cigar in read.cigar:
                # Only count where the read matched to the genome
                if cigar.type not in self.coverage_cigar:
                    continue
                match_start = cigar.ref_iv.start - self.start
                match_stop = cigar.ref_iv.end - self.start

                if match_stop < 0:
                    # If the match_stop is negative, that means we have the
                    # other read of the paired end read that mapped to
                    # somewhere else in the genome
                    continue
                match_start = max(match_start, 0)
                match_stop = min(match_stop, self.length)

                counts[match_start:match_stop] += 1
        if self.log_base is not None:
            counts = np.log(counts)/np.log(self.log_base)
        return counts

    def count_junctions(self):
        """"""
        junctions = Counter()
        region_reads = self.bam[self.interval]

        for read in region_reads:
            if read is None:
                continue
            for cigar in read.cigar:
                # N = did not match to genome and is an insertion, therefore
                # a junction read!
                if cigar.type == 'N':
                    junction_start = cigar.ref_iv.start - self.start

                    junction_stop = junction_start + cigar.ref_iv.length

                    if (junction_stop < 0) or (junction_start > self.length):
                        # If any of the junctions start or end outside of the
                        # region, skip it
                        continue
                    junctions[(junction_start, junction_stop)] += 1
        return junctions

    def plot_coverage(self, color, ax, **kwargs):
        xmax = self.coverage.shape[0]
        xvalues = np.arange(0, xmax)
        ax.set(xlim=(0, xmax))

        return ax.fill_between(xvalues, self.coverage, y2=0, linewidth=0,
                               color=color, **kwargs)

    @staticmethod
    def cubic_bezier(points, t):
        """
        Get points in a cubic bezier.
        """
        p0, p1, p2, p3 = points
        p0 = np.array(p0)
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        return p0 * (1 - t) ** 3 + 3 * t * p1 * (1 - t) ** 2 + \
               3 * t ** 2 * (1 - t) * p2 + t ** 3 * p3

    def plot_junctions(self, ax, curve_height_multiplier, text_kws, patch_kws):
        junction_area_counter = np.zeros(self.length)
        for (start, stop), n_junction_reads in self.junctions.items():
            left = max(start, 0)
            right = min(stop, self.length)
            print(left, right)
            voffset = np.max(junction_area_counter[left:right])
            self._plot_single_junction(start, stop, left, right,
                                       n_junction_reads, ax,
                                       curve_height_multiplier,
                                       text_kws=text_kws, patch_kws=patch_kws,
                                       voffset=voffset)
            junction_area_counter[left:right] += 1

    def _plot_single_junction(self, start, stop, left, right, n_reads, ax=None,
                              curve_height_multiplier=0.1, text_kws=TEXT_KWS,
                              patch_kws=PATCH_KWS, voffset=0):
        """Draw a curved cubic bezier line showing the number of junctions

        Uses the y-axis limits to determine the curve height so make sure to
        use this function AFTER plot_coverage

        Parameters
        ----------
        left, right : int
            0-based Left and right integers indicating where the junction
            starts in this region. Not genome coordinates but the subset that
            we care about
        n_reads : int
            Number of junction reads observed on this junction
        curve_height_multiplier : float
            When making the rectangle boundaries for the bezier curve, this
            indicates the percentage of the plot height to use as the curve
        text_kws : dict
            Keyword arguments to pass to the plt.text annotator
        patch_kws : dict
            Keyword arguments to pass to the patch artist for creating the
            curve
        voffset : int
            Vertical offset, for when there are multiple junctions overlapping
            in one area. Increases the starting height of the junction by one
            curve height (found by multiplying ``curve_height_multiplier`` by
            the y-axis range)
        """
        if ax is None:
            ax = plt.gca()

        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        curve_height = yrange * curve_height_multiplier

        left_height = self.coverage[left-1] + voffset * curve_height
        right_height = self.coverage[right+1] + voffset * curve_height

        if start < 0:
            left_height = right_height
        if stop > self.length:
            right_height = left_height

        # Bezier curves are defined by 4 points indicating the rectangle that
        # bounds the curve
        vertices = [(left, left_height),
                    (left, left_height + curve_height),
                    (right, right_height + curve_height),
                    (right, right_height)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        midpoint = self.cubic_bezier(vertices, 0.5)

        if n_reads:
            plt.text(midpoint[0], midpoint[1], '{}'.format(n_reads),
                     **text_kws)

        path = Path(vertices, codes)

        patch_kws['linewidth'] = np.log(n_reads + 1) / np.log(self.log_base)
        if self.color is not None:
            patch_kws['edgecolor'] = self.color
        patch = PathPatch(path, **patch_kws)
        return ax.add_patch(patch)


def wasabiplot(bam_filename, chrom, start, stop, strand, log_base=10,
               color='steelblue', bad_cigar=INSERTION_DELETIONS,
               coverage_cigar=COVERAGE_CIGAR, junction_cigar=JUNCTION_CIGAR,
               ax=None, coverage_kws=None, curve_height_multiplier=0.2,
               text_kws=TEXT_KWS, patch_kws=PATCH_KWS, warn_skipped=True,
               **kwargs):
    """Get the number of reads that matched to the reference sequence

    Parameters
    ----------
    bam_filename : str
        Name of the bam filename for logging purposes
    chrom : str
        Name of the reference chromosome
    start, stop : int
        Genome-based locations of the start and stop regions
    strand : '+' | '-'
        Strand to query
    log_base : number or None, optional
        The base to use for log-scaling the data. e.g. 10 would have log10 data
        If None, the data is not log-scaled. (default=10)
    color : valid matplotlib color
        Color to use for both the coverage and junction plotting

    allowed_cigar : tuple of str, optional
        Which CIGAR string flags are allowed. (default=('M') aka match)
    bad_cigar : tuple of str, optional
        Which CIGAR string flags are not allowed. (default=('I', 'D') aka
        insertion and deletion)

    """
    if isinstance(bam_filename, pd.Series):
        bam_filename = bam_filename.iloc[0]

    plotter = WasabiPlotter(bam_filename, chrom, start, stop, strand, log_base,
                            color, bad_cigar, coverage_cigar, junction_cigar,
                            warn_skipped)

    if ax is None:
        ax = plt.gca()

    coverage_kws = {} if coverage_kws is None else coverage_kws
    coverage_kws.update(kwargs)

    plotter.plot_coverage(color, ax, **coverage_kws)
    plotter.plot_junctions(ax, curve_height_multiplier=curve_height_multiplier,
                           text_kws=text_kws, patch_kws=patch_kws)
    if log_base is not None:
        yticks = [int(ytick) for ytick in ax.get_yticks()]
        yticklabels = ['${log_base}^{{{exponent}}}$'.format(
            log_base=log_base, exponent=ytick) for ytick in yticks]
        ax.set(yticklabels=yticklabels, yticks=yticks)

    if ax.is_last_row():
        xticks = [int(x + start) for x in ax.get_xticks()]
        xlabel = '{chrom}:{start}-{stop}:{strand}'.format(
            chrom=chrom, start=start, stop=stop, strand=strand)
        ax.set(xticklabels=xticks, xlabel=xlabel)

"""Export four full-trajectory examples per maneuver style to a PDF.

Usage:
    python scripts/plot_style_examples.py
    python scripts/plot_style_examples.py --styles half_moon circle --seq-len 160
Python:
    from scripts.plot_style_examples import generate_style_pdf
    generate_style_pdf(styles=['half_moon', 'circle'])
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Trajectory_Gen import MANEUVER_STYLES, generate_single_style_trajectory


def generate_style_pdf(output=None, styles=None, seq_len=160, delta_t=0.1, seed=42):
    """Return the PDF path. One page/style, four seeded parameter variants/page.

    Plots full source trajectories, not training windows. Gray tails indicate
    straight_line, circle or helix_up after the finite maneuver has completed.
    """
    styles = list(MANEUVER_STYLES if styles is None else styles)
    if not styles or any(style not in MANEUVER_STYLES for style in styles):
        raise ValueError('styles must be a nonempty selection from MANEUVER_STYLES')
    if seq_len < 2 or delta_t <= 0:
        raise ValueError('seq_len >= 2 and delta_t > 0 are required')
    output = Path(output) if output is not None else ROOT / 'output/pdf/style_examples.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reusing these parameters across styles makes comparisons reproducible.
    rng = np.random.default_rng(seed)
    variants = [(r + rng.uniform(-0.15, 0.15), h + rng.uniform(-0.3, 0.3), d)
                for r, h, d in [(2, 8, 5), (3.5, 10, 6), (5, 12, 7), (6, 15, 8)]]
    tails = {'split_s':'straight_line', 'immelmann':'straight_line',
             'spiral_inward':'circle', 'spiral_outward':'circle', 'conical_helix':'helix_up'}
    colors = ['#277DA1', '#E07A37', '#278C75', '#8B65A8']
    with plt.rc_context({'font.family':'DejaVu Sans', 'font.size':9, 'pdf.fonttype':42}):
        with PdfPages(output) as pdf:
            pdf.infodict()['Title'] = 'Trajectory style examples'
            for page, style in enumerate(styles, 1):
                fig = plt.figure(figsize=(10, 8))
                fig.suptitle(style.replace('_',' ').title(), x=.07, y=.97, ha='left', fontsize=20, weight='bold')
                fig.text(.07,.922, f'Full trajectories  |  {seq_len} frames  |  dt = {delta_t:g} s  |  seed = {seed}', color='#596579')
                for j,(radius,height,duration) in enumerate(variants):
                    xyz = generate_single_style_trajectory(style,seq_len,height,radius,delta_t,duration)
                    assert np.isfinite(xyz).all()
                    ax = fig.add_subplot(2,2,j+1,projection='3d')
                    ax.plot(*xyz.T, color=colors[j],lw=2)
                    if style in tails:
                        start = np.searchsorted(np.arange(seq_len)*delta_t,duration)
                        if start < seq_len:
                            ax.plot(*xyz[start:].T,color='#7D8794',lw=2)
                    ax.scatter(*xyz[0],color='#182230',s=23,depthshade=False)
                    ax.scatter(*xyz[-1],color=colors[j],marker='*',s=80,edgecolor='white',linewidth=.4,depthshade=False)
                    timing = f'Half-circle duration {(seq_len-1)*delta_t:g} s' if style == 'half_moon' else f'Duration parameter {duration:g} s'
                    ax.set_title(f'{j+1}.  Radius {radius:.2f} m  |  Height {height:.2f} m\n{timing}',loc='left',fontsize=10)
                    ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
                    lo,hi=xyz.min(0),xyz.max(0)
                    span=np.maximum(hi-lo,1.0)
                    mid=(hi+lo)/2
                    for setter,m,w in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],mid,span):
                        setter(m-.55*w,m+.55*w)
                    # Equal physical scale without wasting space on flat trajectories.
                    ax.set_box_aspect(span)
                    ax.view_init(elev=25,azim=-55)
                    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                        axis.set_major_locator(MaxNLocator(nbins=3))
                    ax.tick_params(labelsize=7,pad=1)
                    ax.xaxis.labelpad=3; ax.yaxis.labelpad=3; ax.zaxis.labelpad=3
                legend=[Line2D([],[],color='#182230',marker='o',lw=0,label='Start'),
                        Line2D([],[],color=colors[0],marker='*',lw=0,markersize=9,label='End')]
                if style in tails:
                    legend.append(Line2D([],[],color='#7D8794',lw=2,label=f'Tail: {tails[style]}'))
                fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.03),ncol=3,frameon=False)
                note = 'Half Moon spans 180 degrees over the full sequence; peak height increase = 0.8 x radius.' if style=='half_moon' else 'Duration parameter controls maneuver phase; axes use equal physical scale within each panel.'
                fig.text(.07,.014,note,fontsize=8,color='#596579')
                fig.text(.97,.014,f'{page}/{len(styles)}',ha='right',fontsize=8)
                fig.subplots_adjust(left=.04,right=.95,bottom=.12,top=.85,wspace=.12,hspace=.32)
                pdf.savefig(fig)
                plt.close(fig)
    return output.resolve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--styles', nargs='+', choices=MANEUVER_STYLES)
    parser.add_argument('--seq-len',type=int,default=160)
    parser.add_argument('--delta-t',type=float,default=.1)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    print(generate_style_pdf(args.output,args.styles,args.seq_len,args.delta_t,args.seed))

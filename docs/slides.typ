#import "@preview/touying:0.6.1": *
#import "@local/lilka:0.0.0": *
#import "@preview/simplebnf:0.1.1": *
#import "@preview/treet:0.1.1": *

#show: lilka


#let title = "btc2sim"
#let info = (author: "Noah Syrkis", date: datetime.today(), title: title)
#show: slides.with(config-info(..info), config-common(handout: false))
#metadata((title: title, slug: "miiii"))

#title-slide()



= Grammar

#figure(
  bnf(
    Prod(
      delim: $→$,

      emph[root],
      {
        Or[#emph[tree] $(triangle.stroked.r$ #emph[tree])$star$][program]
      },
    ),
    Prod(
      delim: $→$,
      emph[tree],
      {
        Or[#emph[leaf] | #emph[node]][node or leaf]
      },
    ),
    Prod(
      delim: $→$,
      emph[leaf],
      {
        Or[A (#emph[move] | #emph[stand])][action]
        Or[C (is_alive | in_range)][condition]
      },
    ),
    Prod(
      delim: $→$,
      emph[node],
      {
        Or[S (#emph[root])][sequence]
        Or[F (#emph[root])][fallback]
      },
    ),
    Prod(
      delim: $→$,
      emph[move],
      {
        Or[#emph[move direction]][move action]
      },
    ),
    Prod(
      delim: $→$,
      emph[direction],
      {
        Or[to | from][direction]
      },
    ),
  ),
  caption: [The BTC2SIM DSL grammar],
)

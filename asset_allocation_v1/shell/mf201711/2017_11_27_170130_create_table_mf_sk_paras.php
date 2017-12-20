<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateTableMfSkParas extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('mf_sk_paras', function($table) {
	    $table->increments('id');
	    $table->string('mf_id')->comment('多因子策略id,0*A股,1*A股基金,2*债券,3*债券基金,4*港股,5*港股基金,6*美股,7*美股基金');
	    $table->integer('if_st')->unsigned()->comment('是否过滤ST及*ST股票');
	    $table->integer('layer_num')->unsigned()->comment('股票分层数');
	    $table->integer('lookback_num')->unsigned()->comment('取均值回溯期限');
	    $table->integer('if_classify')->unsigned()->comment('0:不进行大类初筛, 1:进行大类初筛，每类保留1个因子，以此类推');
	    $table->integer('use_factor_method')->unsigned()->comment('0:按名次取因子, 1:按百分位取因子');
	    $table->integer('use_factor_num')->unsigned()->comment('取用前几名/多少百分位以上因子');
	    $table->date('start_time')->default('0000-00-00')->comment('策略开始时间');
	    $table->timestamps();
	});
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::drop('mf_sk_paras');
    }
}
